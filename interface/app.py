"""
Interface web simples pra testar o projeto:
  - aba Treinar: envia um video, importa pro dataset e treina YOLO, RF-DETR ou os dois
  - aba Testar: roda o modelo escolhido numa imagem, num video, na webcam ou na ESP32

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
    sys.exit(subprocess.call(
        [PYTHON, os.path.abspath(__file__)] + sys.argv[1:],
        env=dict(os.environ, LARA_RELANCADO="1"),
    ))

import base64
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

app = Flask(__name__)


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


@app.post("/enviar")
def enviar():
    if estado["rodando"]:
        return jsonify(erro="ja tem um processo rodando"), 409

    video = request.files.get("video")
    produto = (request.form.get("produto") or "").strip()
    fundo = request.form.get("fundo") == "true"
    preview = request.form.get("preview") == "true"
    opcoes, erro = lerOpcoesTreino()

    if not video or not video.filename:
        return jsonify(erro="envie um video"), 400
    if not fundo and not produto:
        return jsonify(erro="informe o nome do produto"), 400
    if erro:
        return jsonify(erro=erro), 400

    os.makedirs(PASTA_UPLOADS, exist_ok=True)
    nome = f"{int(time.time())}_{os.path.basename(video.filename)}"
    caminho = os.path.join(PASTA_UPLOADS, nome)
    video.save(caminho)

    comando = [PYTHON, "-u", SCRIPT, caminho, "--fundo" if fundo else produto, "--sem-treino"]
    if preview:
        comando.append("--preview")

    etapas = [("importar video", "extraindo frames", comando)] + etapasDeTreino(*opcoes)
    comecar(etapas, "extraindo frames")
    return jsonify(ok=True)


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
    deteccoes.sort(key=lambda d: d["conf"], reverse=True)
    imagem = "data:image/jpeg;base64," + base64.b64encode(jpeg(resultado.plot(), 90)).decode()
    return jsonify(imagem=imagem, deteccoes=deteccoes, tempo_ms=tempo_ms)


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
        try:
            while True:
                inicio = time.time()
                ok, frame = captura.read()
                if not ok:
                    break
                resultado = modelo(frame, conf=conf, verbose=False)[0]
                saida = resultado.plot()

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
