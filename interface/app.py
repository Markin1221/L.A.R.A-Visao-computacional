"""
Interface web simples pra testar o projeto:
  - aba Treinar: envia um video, importa pro dataset e treina YOLO, RF-DETR ou os dois
  - aba Testar: roda o modelo escolhido numa imagem, num video, na webcam ou na ESP32
  - profundidade: conta tambem os produtos escondidos atras do primeiro da fileira (secao "profundidade" deste arquivo)
  - MQTT: publica as deteccoes e a contagem de cada produto (estoque), ver interface/mqtt_lara.py

Pro RF-DETR, instale no mesmo venv:  venv\\Scripts\\pip install "rfdetr[train]"

Uso:
    venv\\Scripts\\pip install flask
    python interface/app.py
    abra http://127.0.0.1:5000
"""
import os
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# usa sempre o python do venv do projeto (onde estao opencv/ultralytics/torch);
# se foi aberto com outro python, reinicia sozinho com o do venv
_venv_win = os.path.join(RAIZ, "venv", "Scripts", "python.exe")
_venv_unix = os.path.join(RAIZ, "venv", "bin", "python")
PYTHON = next((p for p in (_venv_win, _venv_unix) if os.path.exists(p)), sys.executable)

if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(PYTHON)) \
        and not os.environ.get("LARA_RELANCADO"):
    print(f"reiniciando com o python do venv: {PYTHON}")
    try:
        sys.exit(subprocess.call(
            [PYTHON, os.path.abspath(__file__)] + sys.argv[1:],
            env=dict(os.environ, LARA_RELANCADO="1"),
        ))
    except KeyboardInterrupt:   # Ctrl+C: sai quieto, sem traceback
        sys.exit(0)

import base64
import json
import importlib.util
import re
import threading
import types
import time
import urllib.request
from datetime import datetime

import cv2 as cv
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from mqtt_lara import LaraMQTT

SCRIPT = os.path.join(RAIZ, "testes-b-box", "popular_dataset.py")
TREINO_YOLO = os.path.join(RAIZ, "testes-b-box", "treinar.py")
PASTA_RF = os.path.join(RAIZ, "rf-detr")
TREINO_RF = os.path.join(PASTA_RF, "treinar.py")
MODELO_RF = os.path.join(PASTA_RF, "modelos", "melhor.pth")

# o RF-DETR treina com o python deste venv se ele tiver o rfdetr; senao, com o venv do rf-detr/
_rf_venv = next((p for p in (os.path.join(PASTA_RF, "venv", "Scripts", "python.exe"),
                             os.path.join(PASTA_RF, "venv", "bin", "python")) if os.path.exists(p)), None)
RF_AQUI = importlib.util.find_spec("rfdetr") is not None
PYTHON_RF = PYTHON if RF_AQUI or not _rf_venv else _rf_venv
NOMES_MODELO = {"yolo": "YOLO", "rfdetr": "RF-DETR"}
PASTA_UPLOADS = os.path.join(RAIZ, "uploads")
PASTA_TESTES = os.path.join(PASTA_UPLOADS, "testes")
MODELO = os.path.join(RAIZ, "modelos", "melhor.pt")
MAX_LINHAS = 600

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LINHA_EPOCA = re.compile(r"^\s*(\d+)/(\d+)\s")      # ex: "  3/50   1.2G  1.234 ..."
BARRA = re.compile(r"\d+%\||it/s|s/it|\d+/\d+\s*\[")

# ---------------------------------------------------------------- profundidade
# Profundidade pra contar o que esta ESCONDIDO atras do primeiro produto da fileira (gondola de supermercado).
#
# Ideia (camera de frente, produtos encostados no fundo da prateleira):
#   - o detector (YOLO / RF-DETR) acha o produto da FRENTE de cada fileira;
#   - um modelo de profundidade (Depth Anything V2) diz o quanto esse produto esta RECUADO
#     em relacao a beirada da prateleira;
#   - quantidade na fileira = (profundidade da prateleira - recuo) / profundidade do produto.
#     Fileira cheia: o da frente fica na beirada (recuo 0). Sobrou 1: ele fica la no fundo.
#
# O modelo de profundidade devolve uma profundidade RELATIVA (sem escala). Por isso a calibracao:
# voce marca na imagem um ponto na BEIRADA da prateleira e um no FUNDO (parede do fundo, num vao)
# e diz as distancias reais. Com esses dois pontos de referencia, em todo frame, da pra converter
# a profundidade relativa em centimetros. A camera precisa ficar parada depois de calibrar.
#
# Precisa de: pip install transformers   (o torch ja vem com o ultralytics)
# Na primeira vez baixa o modelo (uns 100 MB) do Hugging Face.

CAMINHO_CAL = os.path.join(RAIZ, "modelos", "calibracao_prateleira.json")
MODELO_PROF = os.environ.get("LARA_PROFUNDIDADE_MODELO", "depth-anything/Depth-Anything-V2-Small-hf")
INTERVALO_AO_VIVO = 0.6   # s entre mapas de profundidade no ao vivo (a camera fica parada)

PADRAO = {
    "frente": None,                # [x, y] de 0 a 1: um ponto na beirada da prateleira
    "fundo": None,                 # [x, y] de 0 a 1: um ponto na parede do fundo
    "dist_camera_cm": 60.0,        # da camera ate a beirada da prateleira
    "prof_prateleira_cm": 40.0,    # da beirada ate o fundo
    "prof_padrao_cm": 8.0,         # profundidade de um produto (quando a classe nao tem a sua)
    "prof_produto_cm": {},         # {"doritos": 6, "coca_cola_2l": 11}
}


def carregarCalibracao():
    cal = dict(PADRAO)
    try:
        with open(CAMINHO_CAL, encoding="utf-8") as f:
            cal.update(json.load(f))
    except (OSError, ValueError):
        pass
    return cal


class Profundidade:
    def __init__(self):
        self.cal = carregarCalibracao()
        self.modelo = self.processador = self.torch = self.dispositivo = None
        self.erro = None
        self.trava = threading.Lock()
        self.ultimo_mapa = None
        self.ultimo_em = 0.0

    # ------------------------------------------------------------ estado
    def disponivel(self):
        return importlib.util.find_spec("transformers") is not None

    def calibrada(self):
        return bool(self.cal.get("frente") and self.cal.get("fundo"))

    def info(self):
        return {"disponivel": self.disponivel(), "carregado": self.modelo is not None, "erro": self.erro,
                "calibrada": self.calibrada(), "calibracao": self.cal, "modelo": MODELO_PROF}

    def salvar(self, novo):
        cal = dict(self.cal)
        for chave in PADRAO:
            if chave in novo:
                cal[chave] = novo[chave]
        for chave in ("dist_camera_cm", "prof_prateleira_cm", "prof_padrao_cm"):
            cal[chave] = max(float(cal[chave]), 0.5)
        cal["prof_produto_cm"] = {str(k): max(float(v), 0.5) for k, v in (cal.get("prof_produto_cm") or {}).items()
                                  if v not in (None, "")}
        for chave in ("frente", "fundo"):
            p = cal.get(chave)
            cal[chave] = [min(max(float(p[0]), 0), 1), min(max(float(p[1]), 0), 1)] if p else None
        os.makedirs(os.path.dirname(CAMINHO_CAL), exist_ok=True)
        with open(CAMINHO_CAL, "w", encoding="utf-8") as f:
            json.dump(cal, f, ensure_ascii=False, indent=2)
        self.cal = cal
        return cal

    # ------------------------------------------------------------ modelo
    def _carregar(self):
        if self.modelo is not None:
            return
        if not self.disponivel():
            self.erro = "instale com: venv\\Scripts\\pip install transformers"
            raise RuntimeError(self.erro)
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            print(f"carregando o modelo de profundidade {MODELO_PROF} (na primeira vez baixa uns 100 MB)...")
            self.torch = torch
            self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
            self.processador = AutoImageProcessor.from_pretrained(MODELO_PROF)
            self.modelo = AutoModelForDepthEstimation.from_pretrained(MODELO_PROF).to(self.dispositivo).eval()
            self.erro = None
        except Exception as e:
            self.erro = f"nao carregou o modelo de profundidade: {e}"
            raise RuntimeError(self.erro) from e

    def mapa(self, frame):
        """Profundidade relativa (maior = mais PERTO), do tamanho do frame."""
        with self.trava:
            self._carregar()
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            entrada = self.processador(images=rgb, return_tensors="pt").to(self.dispositivo)
            with self.torch.no_grad():
                saida = self.modelo(**entrada).predicted_depth
            m = saida.squeeze().float().cpu().numpy()
        return cv.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_LINEAR)

    def mapaAoVivo(self, frame, intervalo=INTERVALO_AO_VIVO):
        """No ao vivo recalcula so de tempos em tempos (a camera e a gondola ficam paradas)."""
        agora = time.time()
        if self.ultimo_mapa is None or self.ultimo_mapa.shape != frame.shape[:2] or \
                agora - self.ultimo_em >= intervalo:
            self.ultimo_mapa, self.ultimo_em = self.mapa(frame), agora
        return self.ultimo_mapa

    def reiniciarAoVivo(self):
        self.ultimo_mapa = None

    # ------------------------------------------------------------ contagem
    @staticmethod
    def _amostra(mapa, ponto, raio=0.015):
        h, w = mapa.shape
        x, y = int(ponto[0] * w), int(ponto[1] * h)
        r = max(int(raio * max(w, h)), 3)
        regiao = mapa[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        return float(np.median(regiao)) if regiao.size else float(mapa[min(y, h - 1), min(x, w - 1)])

    def _escala(self, mapa):
        """
        O modelo da a profundidade 'invertida' a menos de escala e deslocamento: d = s / z + t.
        Com a beirada (z1) e o fundo (z2) como referencias, acha s e t neste frame.
        """
        z1 = float(self.cal["dist_camera_cm"])
        z2 = z1 + float(self.cal["prof_prateleira_cm"])
        d1, d2 = self._amostra(mapa, self.cal["frente"]), self._amostra(mapa, self.cal["fundo"])
        if d1 - d2 <= 1e-6:   # o fundo "mais perto" que a frente: pontos marcados errado
            return None
        s = (d1 - d2) / (1 / z1 - 1 / z2)
        return s, d1 - s / z1, z1

    def estimar(self, mapa, caixas, classes):
        """
        caixas: Nx4 (x0, y0, x1, y1) em px; classes: nomes. Devolve, por caixa,
        {"quantidade": n, "recuo_cm": cm ou None}. Sem calibracao, conta 1 por caixa.
        """
        escala = self._escala(mapa) if self.calibrada() else None
        prof_prat = float(self.cal["prof_prateleira_cm"])
        saida = []
        for (x0, y0, x1, y1), classe in zip(caixas, classes):
            if escala is None:
                saida.append({"quantidade": 1, "recuo_cm": None})
                continue
            s, t, z1 = escala
            # meio da caixa: a face da frente do produto (as bordas pegam o fundo ou o vizinho)
            cx0, cx1 = int(x0 + (x1 - x0) * 0.3), int(x0 + (x1 - x0) * 0.7)
            cy0, cy1 = int(y0 + (y1 - y0) * 0.3), int(y0 + (y1 - y0) * 0.7)
            regiao = mapa[max(cy0, 0):max(cy1, cy0 + 1), max(cx0, 0):max(cx1, cx0 + 1)]
            d = float(np.median(regiao)) if regiao.size else t + s / z1
            z = s / (d - t) if d - t > 1e-6 else z1 + prof_prat
            recuo = min(max(z - z1, 0.0), prof_prat)
            prof_item = float(self.cal["prof_produto_cm"].get(classe, self.cal["prof_padrao_cm"]))
            n = max(1, int(round((prof_prat - recuo) / prof_item)))
            saida.append({"quantidade": n, "recuo_cm": round(recuo, 1)})
        return saida

    @staticmethod
    def relativa(mapa, caixas):
        """Modo teste (sem prateleira): o quanto cada objeto esta PERTO, de 0 a 100% da cena. Sem calibracao."""
        lo, hi = np.percentile(mapa, 2), np.percentile(mapa, 98)
        saida = []
        for x0, y0, x1, y1 in caixas:
            cx0, cx1 = int(x0 + (x1 - x0) * 0.3), int(x0 + (x1 - x0) * 0.7)
            cy0, cy1 = int(y0 + (y1 - y0) * 0.3), int(y0 + (y1 - y0) * 0.7)
            regiao = mapa[max(cy0, 0):max(cy1, cy0 + 1), max(cx0, 0):max(cx1, cx0 + 1)]
            d = float(np.median(regiao)) if regiao.size else lo
            saida.append({"proximidade": int(round(100 * min(max((d - lo) / max(hi - lo, 1e-6), 0), 1)))})
        return saida

    @staticmethod
    def desenharRelativa(img, mapa, caixas, relativas):
        """Imagem | mapa de profundidade, lado a lado (claro = perto). Cada objeto mostra '% perto'."""
        lo, hi = np.percentile(mapa, 2), np.percentile(mapa, 98)
        m = (np.clip((mapa - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        cor = cv.applyColorMap(m, cv.COLORMAP_INFERNO)
        for (x0, y0, x1, y1), e in zip(caixas, relativas):
            for alvo in (img, cor):
                texto = f"{e['proximidade']}% perto"
                pos = (int(x0) + 6, max(int(y0) + 26, 26))
                cv.putText(alvo, texto, pos, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv.LINE_AA)
                cv.putText(alvo, texto, pos, cv.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 160), 2, cv.LINE_AA)
            cv.rectangle(cor, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)
        h = img.shape[0]
        for i, (rot, cl) in enumerate((("perto", 255), ("longe", 0))):
            amostra = cv.applyColorMap(np.array([[cl]], np.uint8), cv.COLORMAP_INFERNO)[0, 0].tolist()
            cv.rectangle(cor, (12, h - 60 + i * 26), (32, h - 42 + i * 26), amostra, -1)
            cv.putText(cor, rot, (40, h - 44 + i * 26), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
        return np.hstack([img, cor])

    # ------------------------------------------------------------ desenho
    def desenhar(self, img, mapa, caixas, estimativas, mostrar_mapa=True):
        h, w = img.shape[:2]
        for (x0, y0, x1, y1), e in zip(caixas, estimativas):
            texto = f"x{e['quantidade']}"
            if e["recuo_cm"] is not None:
                texto += f"  {e['recuo_cm']:.0f}cm"
            pos = (int(x0) + 6, int(y1) - 10)
            cv.putText(img, texto, pos, cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv.LINE_AA)
            cv.putText(img, texto, pos, cv.FONT_HERSHEY_SIMPLEX, 0.9, (80, 255, 160), 2, cv.LINE_AA)
        for chave, cor in (("frente", (80, 255, 160)), ("fundo", (255, 120, 230))):
            p = self.cal.get(chave)
            if p:
                c = (int(p[0] * w), int(p[1] * h))
                cv.circle(img, c, 9, (0, 0, 0), 4, cv.LINE_AA)
                cv.circle(img, c, 9, cor, 2, cv.LINE_AA)
                cv.putText(img, chave, (c[0] + 12, c[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2, cv.LINE_AA)
        if mostrar_mapa and mapa is not None:   # miniatura do mapa de profundidade no canto
            lw = max(w // 4, 80)
            lh = int(lw * h / w)
            m = cv.normalize(mapa, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            mini = cv.applyColorMap(cv.resize(m, (lw, lh)), cv.COLORMAP_INFERNO)
            img[8:8 + lh, w - lw - 8:w - 8] = mini
            cv.rectangle(img, (w - lw - 8, 8), (w - 8, 8 + lh), (255, 255, 255), 1)
        if not self.calibrada():
            msg = "profundidade sem calibracao: contando so o que aparece"
            cv.putText(img, msg, (12, h - 16), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(img, msg, (12, h - 16), cv.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 255), 2, cv.LINE_AA)
        return img


app = Flask(__name__)
MQTT = LaraMQTT()   # deteccoes + contagem do estoque (nao trava nada se o broker estiver fora)
PROF = Profundidade()   # modelo de profundidade so carrega quando alguem ligar a opcao


# ---------------------------------------------------------------- treino

trava = threading.Lock()
estado = {}


def resetarEstado():
    estado.clear()
    estado.update(
        rodando=False, fase="parado", codigo=None, linhas=[],
        epoca=0, total=0, inicio=None, inicio_epoca=None, duracoes=[],
        etapa=0, etapas=0, nome_etapa="", resumo=[],
    )


def zerarEpocas():
    estado.update(epoca=0, total=0, inicio_epoca=None, duracoes=[])


resetarEstado()


def eh_barra(linha):
    return bool(BARRA.search(linha))


def adicionarLinha(linha):
    linha = ANSI.sub("", linha).rstrip()
    if not linha.strip():
        return

    with trava:
        linhas = estado["linhas"]
        # barras de progresso chegam varias vezes; troca a ultima em vez de empilhar
        if linhas and eh_barra(linha) and eh_barra(linhas[-1]) and linha.split()[0] == linhas[-1].split()[0]:
            linhas[-1] = linha
        else:
            linhas.append(linha)
            del linhas[:-MAX_LINHAS]

        if "dataset insuficiente" in linha:
            estado["fase"] = "dataset insuficiente"

        m = LINHA_EPOCA.match(linha)
        if m:
            epoca, total = int(m.group(1)), int(m.group(2))
            agora = time.time()
            if epoca != estado["epoca"]:
                if estado["inicio_epoca"] is not None and epoca > estado["epoca"]:
                    estado["duracoes"].append(agora - estado["inicio_epoca"])
                estado["inicio_epoca"] = agora
                estado["epoca"], estado["total"] = epoca, total
            estado["fase"] = f"treinando {estado['nome_etapa']}"


def executar(etapas):
    """Roda as etapas em sequencia (importar video, treinar YOLO, treinar RF-DETR); para no primeiro erro."""
    with trava:
        estado["etapas"] = len(etapas)
    codigo = 0
    for i, (nome, fase, comando) in enumerate(etapas, 1):
        with trava:
            zerarEpocas()
            estado.update(etapa=i, nome_etapa=nome, fase=fase)
            estado["linhas"].append(f"===== etapa {i} de {len(etapas)}: {nome} =====")
        inicio = time.time()
        codigo = rodarComando(comando)
        with trava:
            estado["resumo"].append({"nome": nome, "segundos": time.time() - inicio, "ok": codigo == 0})
            insuficiente = estado["fase"] == "dataset insuficiente"
        if codigo != 0 or insuficiente:
            break

    with trava:
        estado["rodando"] = False
        estado["codigo"] = codigo
        if codigo != 0:
            estado["fase"] = "erro"
        elif estado["fase"] != "dataset insuficiente":
            estado["fase"] = "concluido"


def rodarComando(comando):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(
        comando, cwd=RAIZ, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    buffer = b""
    while True:
        pedaco = proc.stdout.read1(4096)
        if not pedaco:
            break
        buffer += pedaco
        # tqdm usa \r pra redesenhar a barra; trata como quebra de linha
        partes = re.split(rb"[\r\n]", buffer)
        buffer = partes.pop()
        for p in partes:
            adicionarLinha(p.decode("utf-8", errors="replace"))
    if buffer:
        adicionarLinha(buffer.decode("utf-8", errors="replace"))
    return proc.wait()


def etapasDeTreino(modelo, epocas, tamanho_rf):
    etapas = []
    if modelo in ("yolo", "ambos"):
        etapas.append(("YOLO", "preparando YOLO", [PYTHON, "-u", TREINO_YOLO, "--epocas", epocas]))
    if modelo in ("rfdetr", "ambos"):
        etapas.append(("RF-DETR", "preparando RF-DETR",
                       [PYTHON_RF, "-u", TREINO_RF, "--epocas", epocas, "--tamanho", tamanho_rf]))
    return etapas


def lerOpcoesTreino():
    modelo = request.form.get("modelo") or "yolo"
    epocas = request.form.get("epocas") or "50"
    tamanho_rf = request.form.get("tamanho_rf") or "nano"
    if modelo not in ("yolo", "rfdetr", "ambos"):
        return None, "modelo invalido"
    if not epocas.isdigit() or int(epocas) < 1:
        return None, "epocas invalidas"
    if tamanho_rf not in ("nano", "small", "medium", "large"):
        return None, "tamanho do RF-DETR invalido"
    return (modelo, epocas, tamanho_rf), None


def comecar(etapas, fase):
    with trava:
        resetarEstado()
        estado.update(rodando=True, fase=fase, inicio=time.time())
    threading.Thread(target=executar, args=(etapas,), daemon=True).start()


def calcularEta():
    total, epoca, duracoes = estado["total"], estado["epoca"], estado["duracoes"]
    if not total or not duracoes:
        return None
    media = sum(duracoes) / len(duracoes)
    decorrido_atual = time.time() - estado["inicio_epoca"]
    return max(0, media * (total - epoca + 1) - decorrido_atual)


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/classes")
def classes_existentes():
    """Tags que ja existem no dataset (sugestoes no campo de tag)."""
    try:
        with open(os.path.join(RAIZ, "dataset", "classes.json"), encoding="utf-8") as f:
            return jsonify(classes=sorted(json.load(f)))
    except (OSError, ValueError):
        return jsonify(classes=[])


@app.post("/enviar")
def enviar():
    """
    Recebe um ou varios videos. Cada video tem a sua tag (videos com a mesma tag viram a mesma classe)
    ou e marcado como 'so fundo'. Importa um por um e depois treina uma vez so.
    """
    if estado["rodando"]:
        return jsonify(erro="ja tem um processo rodando"), 409

    videos = [v for v in request.files.getlist("videos") if v and v.filename]
    if videos:
        tags = request.form.getlist("tags")
        fundos = [f == "true" for f in request.form.getlist("fundos")]
    else:   # formato antigo: um video so
        unico = request.files.get("video")
        videos = [unico] if unico and unico.filename else []
        tags = [request.form.get("produto") or ""]
        fundos = [request.form.get("fundo") == "true"]
    preview = request.form.get("preview") == "true"
    opcoes, erro = lerOpcoesTreino()

    if not videos:
        return jsonify(erro="envie pelo menos um video"), 400
    if len(tags) != len(videos) or len(fundos) != len(videos):
        return jsonify(erro="cada video precisa de uma tag (ou ser marcado como fundo)"), 400
    tags = [t.strip() for t in tags]
    sem_tag = [v.filename for v, t, f in zip(videos, tags, fundos) if not f and not t]
    if sem_tag:
        return jsonify(erro="falta a tag de: " + ", ".join(sem_tag)), 400
    if erro:
        return jsonify(erro=erro), 400

    os.makedirs(PASTA_UPLOADS, exist_ok=True)
    etapas = []
    agora = int(time.time())
    for i, (video, tag, fundo) in enumerate(zip(videos, tags, fundos), 1):
        nome = f"{agora}_{i}_{os.path.basename(video.filename)}"
        caminho = os.path.join(PASTA_UPLOADS, nome)
        video.save(caminho)
        comando = [PYTHON, "-u", SCRIPT, caminho, "--fundo" if fundo else tag, "--sem-treino"]
        if preview:
            comando.append("--preview")
        rotulo = "fundo" if fundo else tag
        titulo = f"importar {i}/{len(videos)} ({rotulo})" if len(videos) > 1 else f"importar video ({rotulo})"
        etapas.append((titulo, "extraindo frames", comando))

    etapas += etapasDeTreino(*opcoes)
    comecar(etapas, "extraindo frames")
    return jsonify(ok=True, videos=len(videos))


@app.post("/treinar")
def treinar_de_novo():
    """Treina com o dataset que ja existe, sem importar video novo."""
    if estado["rodando"]:
        return jsonify(erro="ja tem um processo rodando"), 409
    opcoes, erro = lerOpcoesTreino()
    if erro:
        return jsonify(erro=erro), 400
    pasta = os.path.join(RAIZ, "dataset", "images", "train")
    if not os.path.isdir(pasta) or not os.listdir(pasta):
        return jsonify(erro="o dataset esta vazio: envie um video primeiro"), 400
    etapas = etapasDeTreino(*opcoes)
    comecar(etapas, etapas[0][1])
    return jsonify(ok=True)


@app.get("/status")
def status():
    with trava:
        decorrido = time.time() - estado["inicio"] if estado["inicio"] else 0
        return jsonify(
            rodando=estado["rodando"],
            fase=estado["fase"],
            codigo=estado["codigo"],
            epoca=estado["epoca"],
            total=estado["total"],
            eta=calcularEta() if estado["rodando"] else None,
            decorrido=decorrido,
            etapa=estado["etapa"],
            etapas=estado["etapas"],
            nome_etapa=estado["nome_etapa"],
            resumo=estado["resumo"],
            linhas=estado["linhas"][-400:],
        )


# ---------------------------------------------------------------- teste

trava_modelo = threading.Lock()
cache_modelo = {}   # chave -> (mtime, modelo)

# chave -> (tipo, arquivo de pesos ou None = pre-treinado no COCO, nome pra mostrar)
MODELOS = {
    "treinado": ("yolo", MODELO, None),
    "coco": ("yolo", os.path.join(RAIZ, "testes-yolo", "yolo11n.pt"), None),
    "rfdetr": ("rfdetr", MODELO_RF, None),
    "rfdetr-coco": ("rfdetr", None, "RF-DETR Nano pre-treinado (COCO)"),
    "rfdetr-seg": ("rfdetr-seg", None, "RF-DETR Seg Nano pre-treinado (COCO)"),
}


class ErroModelo(Exception):
    pass


@app.errorhandler(ErroModelo)
def erro_modelo(e):
    return jsonify(erro=str(e)), 503


class ModeloRF:
    """Deixa o RF-DETR com a mesma 'cara' do YOLO do ultralytics: modelo(frame, conf=...)[0].plot()"""

    def __init__(self, pesos, segmentar):
        if PASTA_RF not in sys.path:
            sys.path.insert(0, PASTA_RF)
        import comum   # rf-detr/comum.py
        self.comum = comum
        self.modelo = comum.criarModelo("nano", pesos, segmentar=segmentar)
        nomes = getattr(self.modelo, "class_names", None) or []
        self.names = dict(enumerate(nomes)) if isinstance(nomes, (list, tuple)) else dict(nomes)

    def __call__(self, frame, conf=0.25, verbose=False):
        det = self.comum.preverBGR(self.modelo, frame, conf)
        nomes = self.comum.nomesDasDeteccoes(det, self.modelo)
        caixas = types.SimpleNamespace(cls=np.asarray(det.class_id), conf=np.asarray(det.confidence),
                                       xyxy=np.asarray(det.xyxy))
        comum = self.comum
        return [types.SimpleNamespace(
            boxes=caixas,
            names={int(c): n for c, n in zip(det.class_id, nomes)},
            plot=lambda: comum.desenhar(frame, det, nomes),
        )]


def escolhaModelo():
    chave = request.values.get("modelo", "treinado")
    return (chave, *MODELOS[chave]) if chave in MODELOS else ("treinado", *MODELOS["treinado"])


def carregarModelo():
    """Carrega o modelo escolhido uma vez e recarrega sozinho se ele for re-treinado."""
    chave, tipo, pesos, _ = escolhaModelo()
    if pesos is not None and not os.path.exists(pesos):
        return None
    mtime = os.path.getmtime(pesos) if pesos else 0
    with trava_modelo:
        item = cache_modelo.get(chave)
        if item is None or item[0] != mtime:
            if tipo == "yolo":
                from ultralytics import YOLO
                modelo = YOLO(pesos)
            else:
                if importlib.util.find_spec("rfdetr") is None:
                    raise ErroModelo('RF-DETR nao instalado neste venv. Rode: venv\\Scripts\\pip install "rfdetr[train]"')
                try:
                    modelo = ModeloRF(pesos, segmentar=tipo == "rfdetr-seg")
                except Exception as e:
                    raise ErroModelo(f"nao deu pra carregar o RF-DETR: {e}")
            cache_modelo[chave] = (mtime, modelo)
        return cache_modelo[chave][1]


def lerConf():
    try:
        return min(0.95, max(0.01, float(request.values.get("conf", 0.25))))
    except ValueError:
        return 0.25


def caixasDe(resultado):
    xyxy = resultado.boxes.xyxy
    xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
    return np.asarray(xyxy, float).reshape(-1, 4)


def lerProf():
    """'0' desligada | '1' prateleira (conta os de tras, precisa calibrar) | 'rel' teste (so relativa)."""
    modo = request.values.get("prof", "0")
    return modo if modo in ("1", "rel") else None


def jpeg(frame, qualidade=85):
    return cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, qualidade])[1].tobytes()


@app.route("/testar")
def testar():
    return render_template("testar.html")


@app.route("/config")
def config():
    return render_template("config.html")


@app.get("/modelo")
def modelo_info():
    chave, tipo, pesos, rotulo = escolhaModelo()
    try:
        modelo = carregarModelo()
    except ErroModelo as e:
        return jsonify(existe=False, erro=str(e))
    if modelo is None:
        return jsonify(existe=False, arquivo=os.path.relpath(pesos, RAIZ).replace(os.sep, "/"))
    if pesos:
        atualizado = datetime.fromtimestamp(os.path.getmtime(pesos)).strftime("%d/%m/%Y %H:%M")
        arquivo = os.path.relpath(pesos, RAIZ).replace(os.sep, "/")
    else:
        atualizado, arquivo = None, rotulo
    return jsonify(existe=True, arquivo=arquivo, atualizado=atualizado, classes=list(modelo.names.values()))


@app.post("/testar/imagem")
def testar_imagem():
    modelo = carregarModelo()
    if modelo is None:
        return jsonify(erro="nenhum modelo treinado ainda"), 404

    arquivo = request.files.get("arquivo")
    if not arquivo:
        return jsonify(erro="envie uma imagem"), 400
    frame = cv.imdecode(np.frombuffer(arquivo.read(), np.uint8), cv.IMREAD_COLOR)
    if frame is None:
        return jsonify(erro="nao deu pra ler a imagem"), 400

    inicio = time.time()
    resultado = modelo(frame, conf=lerConf(), verbose=False)[0]
    tempo_ms = (time.time() - inicio) * 1000

    deteccoes = [
        {"id": int(c), "classe": resultado.names[int(c)], "conf": float(p)}
        for c, p in zip(resultado.boxes.cls.tolist(), resultado.boxes.conf.tolist())
    ]
    saida = resultado.plot()
    modo_prof = lerProf()
    if modo_prof:
        try:
            caixas = caixasDe(resultado)
            mapa = PROF.mapa(frame)
        except RuntimeError as e:
            return jsonify(erro=str(e)), 503
        if modo_prof == "rel":   # teste: so quao perto cada coisa esta
            relativas = PROF.relativa(mapa, caixas)
            for d, e in zip(deteccoes, relativas):
                d.update(e)
            saida = PROF.desenharRelativa(saida, mapa, caixas, relativas)
        else:                    # prateleira: quantos tem em cada fileira, contando os escondidos atras
            estimativas = PROF.estimar(mapa, caixas, [d["classe"] for d in deteccoes])
            for d, e in zip(deteccoes, estimativas):
                d.update(e)
            PROF.desenhar(saida, mapa, caixas, estimativas)
        tempo_ms = (time.time() - inicio) * 1000
    deteccoes.sort(key=lambda d: d["conf"], reverse=True)
    contagem = MQTT.publicarDeteccoes(deteccoes, "imagem", ao_vivo=False)
    imagem = "data:image/jpeg;base64," + base64.b64encode(jpeg(saida, 90)).decode()
    return jsonify(imagem=imagem, deteccoes=deteccoes, contagem=contagem, tempo_ms=tempo_ms)


@app.get("/calibracao")
def calibracao_ver():
    info = PROF.info()
    try:
        modelo = carregarModelo()
        info["classes"] = list(modelo.names.values()) if modelo is not None else []
    except ErroModelo:
        info["classes"] = []
    return jsonify(info)


@app.post("/calibracao")
def calibracao_salvar():
    try:
        return jsonify(ok=True, calibracao=PROF.salvar(request.get_json(force=True) or {}))
    except (TypeError, ValueError) as e:
        return jsonify(ok=False, erro=f"valor invalido: {e}"), 400


@app.get("/contagem")
def contagem_atual():
    """Contagem estavel do ao vivo + estado do MQTT (a pagina Testar consulta a cada segundo)."""
    return jsonify(MQTT.info())


@app.post("/testar/video")
def testar_video():
    if carregarModelo() is None:
        return jsonify(erro="nenhum modelo treinado ainda"), 404
    arquivo = request.files.get("arquivo")
    if not arquivo or not arquivo.filename:
        return jsonify(erro="envie um video"), 400

    os.makedirs(PASTA_TESTES, exist_ok=True)
    nome = f"{int(time.time())}_{os.path.basename(arquivo.filename)}"
    arquivo.save(os.path.join(PASTA_TESTES, nome))
    return jsonify(id=nome)


class LeitorMJPEG:
    """
    Le o stream MJPEG da ESP32-CAM direto pelo HTTP (sem depender do FFmpeg do OpenCV).
    Fica lendo numa thread e guarda so o frame mais novo, pra ter pouco atraso.
    Tem a mesma "cara" do cv.VideoCapture (read/get/release).
    """

    def __init__(self, url, timeout=5):
        sem_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.resposta = sem_proxy.open(url, timeout=timeout)
        self.frame = None
        self.novo = threading.Event()
        self.ativo = True
        threading.Thread(target=self._ler, daemon=True).start()

    def _ler(self):
        buffer = b""
        try:
            while self.ativo:
                pedaco = self.resposta.read1(65536)
                if not pedaco:
                    break
                buffer += pedaco
                fim = buffer.rfind(b"\xff\xd9")           # fim do ultimo JPEG completo
                if fim == -1:
                    if len(buffer) > 4_000_000:
                        buffer = b""
                    continue
                inicio = buffer.rfind(b"\xff\xd8", 0, fim)  # inicio desse JPEG
                if inicio != -1:
                    img = cv.imdecode(np.frombuffer(buffer[inicio:fim + 2], np.uint8), cv.IMREAD_COLOR)
                    if img is not None:
                        self.frame = img
                        self.novo.set()
                buffer = buffer[fim + 2:]
        except Exception:
            pass
        finally:
            self.ativo = False
            self.novo.set()

    def read(self):
        if not self.novo.wait(timeout=10):
            return False, None
        self.novo.clear()
        if not self.ativo or self.frame is None:
            return False, None
        return True, self.frame

    def get(self, _prop):
        return 0

    def release(self):
        self.ativo = False
        try:
            self.resposta.close()
        except Exception:
            pass


@app.get("/checar_esp")
def checar_esp():
    """Testa a conexao com a ESP32 antes de abrir o stream, pra mostrar um erro claro."""
    url = request.args.get("url", "")
    if not url.startswith(("http://", "https://")):
        return jsonify(ok=False, erro="endereco invalido")
    try:
        leitor = LeitorMJPEG(url, timeout=4)
    except Exception as e:
        return jsonify(ok=False, erro=f"a ESP32 nao respondeu ({e})")
    ok, _ = leitor.read()
    leitor.release()
    time.sleep(0.5)  # da tempo da ESP32 liberar a conexao (ela so aceita um stream por vez)
    return jsonify(ok=ok, erro=None if ok else "conectou, mas nao chegou nenhuma imagem")


@app.get("/stream")
def stream():
    """Stream MJPEG com as deteccoes desenhadas, frame a frame (video enviado ou webcam)."""
    modelo = carregarModelo()
    if modelo is None:
        return "nenhum modelo treinado", 404

    fonte = request.args.get("fonte", "")
    conf = lerConf()
    usar_prof = lerProf()
    if fonte == "webcam" or fonte.startswith("webcam:"):
        indice = fonte.split(":", 1)[1] if ":" in fonte else "0"
        origem, ao_vivo = int(indice) if indice.isdigit() else 0, True
    elif fonte.startswith(("http://", "https://")):
        origem, ao_vivo = fonte, True          # ESP32-CAM (stream MJPEG pela rede)
    else:
        origem = os.path.join(PASTA_TESTES, os.path.basename(fonte))
        ao_vivo = False
        if not os.path.exists(origem):
            return "video nao encontrado", 404

    if isinstance(origem, str) and origem.startswith(("http://", "https://")):
        try:
            captura = LeitorMJPEG(origem)
        except Exception as e:
            return f"nao deu pra conectar na ESP32: {e}", 502
    elif isinstance(origem, int):
        if PASTA_RF not in sys.path:
            sys.path.insert(0, PASTA_RF)
        from comum import abrirCamera   # no Windows usa DirectShow: enxerga DroidCam, Iriun, OBS...
        captura = abrirCamera(origem)
        if not captura.isOpened():
            return "nao deu pra abrir a camera", 500
        captura.set(cv.CAP_PROP_BUFFERSIZE, 1)
    else:
        captura = cv.VideoCapture(origem)
        if not captura.isOpened():
            return "nao deu pra abrir a fonte", 500
        if ao_vivo:
            captura.set(cv.CAP_PROP_BUFFERSIZE, 1)  # menos atraso: pega sempre o frame mais novo

    def gerar():
        fps_video = captura.get(cv.CAP_PROP_FPS) or 30
        intervalo = 0 if ao_vivo else 1 / fps_video
        fps = 0.0
        nome_fonte = "video" if not ao_vivo else ("webcam" if isinstance(origem, int) else "esp32")
        MQTT.reiniciarAoVivo()
        PROF.reiniciarAoVivo()
        erro_prof = None
        try:
            while True:
                inicio = time.time()
                ok, frame = captura.read()
                if not ok:
                    break
                resultado = modelo(frame, conf=conf, verbose=False)[0]
                saida = resultado.plot()
                deteccoes = [{"classe": resultado.names[int(c)], "conf": float(p)}
                             for c, p in zip(resultado.boxes.cls.tolist(), resultado.boxes.conf.tolist())]
                if usar_prof and erro_prof is None:
                    try:
                        caixas = caixasDe(resultado)
                        # teste com voce na webcam: atualiza quase todo frame (voce se mexe; a gondola nao)
                        mapa = (PROF.mapaAoVivo(frame, 0.1 if usar_prof == "rel" else INTERVALO_AO_VIVO)
                                if ao_vivo else PROF.mapa(frame))
                        if usar_prof == "rel":
                            relativas = PROF.relativa(mapa, caixas)
                            for d, e in zip(deteccoes, relativas):
                                d.update(e)
                            saida = PROF.desenharRelativa(saida, mapa, caixas, relativas)
                        else:
                            estimativas = PROF.estimar(mapa, caixas, [d["classe"] for d in deteccoes])
                            for d, e in zip(deteccoes, estimativas):
                                d.update(e)
                            PROF.desenhar(saida, mapa, caixas, estimativas)
                    except RuntimeError as e:
                        erro_prof = str(e)
                if erro_prof:
                    cv.putText(saida, erro_prof[:70], (12, saida.shape[0] - 16), cv.FONT_HERSHEY_SIMPLEX, 0.55,
                               (60, 60, 255), 2, cv.LINE_AA)
                MQTT.publicarDeteccoes(deteccoes, nome_fonte)

                gasto = time.time() - inicio
                fps = 0.9 * fps + 0.1 * (1 / max(gasto, 1e-3)) if fps else 1 / max(gasto, 1e-3)
                cv.putText(saida, f"{fps:.0f} FPS", (12, 32), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                cv.putText(saida, f"{fps:.0f} FPS", (12, 32), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg(saida) + b"\r\n"

                # video: respeita a velocidade original; webcam: o mais rapido possivel
                espera = intervalo - (time.time() - inicio)
                if espera > 0:
                    time.sleep(espera)
        finally:
            captura.release()   # roda tambem quando a pagina para o stream

    return Response(gerar(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("abra http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
