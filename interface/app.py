"""
Interface web simples pra testar o projeto:
  - aba Treinar: envia um video, importa pro dataset e treina, com logs e estimativa de tempo
  - aba Testar: roda o modelos/melhor.pt numa imagem, num video ou na webcam

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
import re
import threading
import time
import urllib.request
from datetime import datetime

import cv2 as cv
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

SCRIPT = os.path.join(RAIZ, "testes-b-box", "popular_dataset.py")
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
    )


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

        if "imagens salvas" in linha:
            estado["fase"] = "preparando treino"
        elif "dataset insuficiente" in linha:
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
            estado["fase"] = "treinando"


def executar(comando):
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

    codigo = proc.wait()
    with trava:
        estado["rodando"] = False
        estado["codigo"] = codigo
        if codigo != 0:
            estado["fase"] = "erro"
        elif estado["fase"] != "dataset insuficiente":
            estado["fase"] = "concluido"


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
    epocas = request.form.get("epocas") or "50"

    if not video or not video.filename:
        return jsonify(erro="envie um video"), 400
    if not fundo and not produto:
        return jsonify(erro="informe o nome do produto"), 400
    if not epocas.isdigit() or int(epocas) < 1:
        return jsonify(erro="epocas invalidas"), 400

    os.makedirs(PASTA_UPLOADS, exist_ok=True)
    nome = f"{int(time.time())}_{os.path.basename(video.filename)}"
    caminho = os.path.join(PASTA_UPLOADS, nome)
    video.save(caminho)

    comando = [PYTHON, "-u", SCRIPT, caminho]
    if fundo:
        comando.append("--fundo")
    else:
        comando.append(produto)
    comando += ["--epocas", epocas]
    if preview:
        comando.append("--preview")

    with trava:
        resetarEstado()
        estado.update(rodando=True, fase="extraindo frames", inicio=time.time())
    threading.Thread(target=executar, args=(comando,), daemon=True).start()
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
            linhas=estado["linhas"][-400:],
        )


# ---------------------------------------------------------------- teste

trava_modelo = threading.Lock()
cache_modelo = {}   # caminho -> (mtime, modelo)

# "treinado" = o nosso; "coco" = YOLO pre-treinado (80 classes: pessoa, garrafa, celular...)
MODELOS = {
    "treinado": MODELO,
    "coco": os.path.join(RAIZ, "testes-yolo", "yolo11n.pt"),
}


def caminhoModelo():
    return MODELOS.get(request.values.get("modelo", "treinado"), MODELO)


def carregarModelo():
    """Carrega o melhor.pt uma vez e recarrega sozinho se ele for re-treinado."""
    caminho = caminhoModelo()
    if not os.path.exists(caminho):
        return None
    mtime = os.path.getmtime(caminho)
    with trava_modelo:
        item = cache_modelo.get(caminho)
        if item is None or item[0] != mtime:
            from ultralytics import YOLO
            cache_modelo[caminho] = (mtime, YOLO(caminho))
        return cache_modelo[caminho][1]


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
    modelo = carregarModelo()
    if modelo is None:
        return jsonify(existe=False)
    caminho = caminhoModelo()
    atualizado = datetime.fromtimestamp(os.path.getmtime(caminho)).strftime("%d/%m/%Y %H:%M")
    arquivo = os.path.relpath(caminho, RAIZ).replace(os.sep, "/")
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
    if fonte == "webcam":
        origem, ao_vivo = 0, True
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
