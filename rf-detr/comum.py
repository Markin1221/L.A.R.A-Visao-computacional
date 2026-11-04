"""Coisas usadas por todos os scripts do experimento RF-DETR."""
import os
import threading
import urllib.request

import cv2 as cv
import numpy as np

PASTA = os.path.dirname(os.path.abspath(__file__))          # rf-detr/
RAIZ = os.path.dirname(PASTA)                                # raiz do repo
DATASET_ORIGINAL = os.path.join(RAIZ, "dataset")            # dataset do YOLO (nao e alterado)
DATASET_RFDETR = os.path.join(PASTA, "dataset")             # copia no formato que o RF-DETR espera
PASTA_RUNS = os.path.join(PASTA, "runs")
PASTA_MODELOS = os.path.join(PASTA, "modelos")
MELHOR_MODELO = os.path.join(PASTA_MODELOS, "melhor.pth")

TAMANHOS = ("nano", "small", "medium", "large")


def criarModelo(tamanho="nano", pesos=None, segmentar=False, **kwargs):
    """
    Cria o RF-DETR do tamanho pedido. Sem pesos = pre-treinado no COCO (80 classes).
    segmentar=True usa a versao de segmentacao: alem da caixa, devolve a mascara
    (o formato exato do objeto, pixel a pixel).
    """
    import rfdetr

    nome = {"nano": "Nano", "small": "Small", "medium": "Medium", "large": "Large"}.get(tamanho)
    if nome is None:
        raise ValueError(f"tamanho invalido: {tamanho} (use {', '.join(TAMANHOS)})")
    classes = {tamanho: f"RFDETRSeg{nome}" if segmentar else f"RFDETR{nome}"}
    if segmentar and not hasattr(rfdetr, classes[tamanho]):
        if hasattr(rfdetr, "RFDETRSegPreview"):   # versoes antigas so tinham essa
            classes[tamanho] = "RFDETRSegPreview"
        else:
            raise SystemExit("essa versao do rfdetr nao tem segmentacao; rode: pip install -U rfdetr")

    if pesos:
        # do jeito recomendado na doc: o checkpoint guarda arquitetura, resolucao e classes
        if hasattr(rfdetr, "RFDETR") and hasattr(rfdetr.RFDETR, "from_checkpoint"):
            try:   # checkpoint gerado aqui mesmo pelo treinar.py: pode confiar
                return rfdetr.RFDETR.from_checkpoint(pesos, trust_checkpoint=True)
            except TypeError:
                try:
                    return rfdetr.RFDETR.from_checkpoint(pesos)
                except Exception:
                    pass
            except Exception:
                pass
        kwargs["pretrain_weights"] = pesos
    return getattr(rfdetr, classes[tamanho])(**kwargs)


def nomesCOCO():
    """Dicionario id -> nome das classes do COCO (o lugar mudou entre versoes do rfdetr)."""
    for caminho in ("rfdetr.assets.coco_classes", "rfdetr.util.coco_classes"):
        try:
            modulo = __import__(caminho, fromlist=["COCO_CLASSES"])
            return dict(modulo.COCO_CLASSES)
        except Exception:
            continue
    return {}


def nomesDasDeteccoes(det, modelo=None):
    """Nome de cada deteccao: modelo treinado traz em data['class_name']; COCO usa a tabela."""
    if det.data is not None and "class_name" in det.data:
        return [str(n) for n in det.data["class_name"]]
    tabela = getattr(modelo, "class_names", None)
    if isinstance(tabela, (list, tuple)):
        tabela = dict(enumerate(tabela))
    if not isinstance(tabela, dict) or not tabela:
        tabela = nomesCOCO()
    return [str(tabela.get(int(c), int(c))) for c in det.class_id]


def preverBGR(modelo, frame_bgr, conf):
    """Roda o RF-DETR num frame do OpenCV (BGR). O RF-DETR espera RGB."""
    rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    return modelo.predict(rgb, threshold=conf)


CORES = [(255, 42, 4), (235, 219, 11), (243, 243, 243), (183, 223, 0), (104, 31, 17),
         (221, 111, 255), (79, 68, 255), (0, 237, 204), (68, 243, 0), (255, 0, 189)]


def desenhar(frame, det, nomes, caixas=True):
    """
    Desenha as deteccoes no estilo do YOLO, sem depender do supervision.
    Se o modelo devolveu mascaras (segmentacao), pinta o formato do objeto por cima.
    """
    saida = frame.copy()
    mascaras = getattr(det, "mask", None)
    if mascaras is not None and len(mascaras):
        pintura = saida.copy()
        for m, c in zip(mascaras, det.class_id):
            m = m.astype(bool)
            if m.shape != saida.shape[:2]:
                m = cv.resize(m.astype(np.uint8), (saida.shape[1], saida.shape[0]), interpolation=cv.INTER_NEAREST).astype(bool)
            cor = CORES[int(c) % len(CORES)]
            pintura[m] = cor
            contornos, _ = cv.findContours(m.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(saida, contornos, -1, cor, 2)
        saida = cv.addWeighted(pintura, 0.45, saida, 0.55, 0)

    for (x1, y1, x2, y2), c, conf, nome in zip(det.xyxy.astype(int), det.class_id, det.confidence, nomes):
        cor = CORES[int(c) % len(CORES)]
        if caixas:
            cv.rectangle(saida, (x1, y1), (x2, y2), cor, 2)
        texto = f"{nome} {conf:.2f}"
        (tw, th), _ = cv.getTextSize(texto, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv.rectangle(saida, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), cor, -1)
        cv.putText(saida, texto, (x1 + 4, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
    return saida


class LeitorMJPEG:
    """Le o stream da ESP32-CAM (MJPEG por HTTP) guardando so o frame mais novo."""

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
                fim = buffer.rfind(b"\xff\xd9")
                if fim == -1:
                    if len(buffer) > 4_000_000:
                        buffer = b""
                    continue
                inicio = buffer.rfind(b"\xff\xd8", 0, fim)
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

    def release(self):
        self.ativo = False
        try:
            self.resposta.close()
        except Exception:
            pass


def abrirCamera(indice):
    """
    Abre a camera N. No Windows tenta primeiro o DirectShow: e por ele que aparecem as
    cameras virtuais (DroidCam, Iriun, OBS...), que o backend padrao (MSMF) nao enxerga.
    """
    import sys
    backends = [cv.CAP_DSHOW, cv.CAP_MSMF] if sys.platform.startswith("win") else [cv.CAP_ANY]
    for backend in backends:
        cap = cv.VideoCapture(indice, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return cv.VideoCapture(indice)


def abrirFonte(fonte):
    """
    'webcam', numero da camera (0, 1...), caminho de video, IP da ESP32 (192.168.1.6)
    ou endereco completo de um stream MJPEG, como o do celular (http://192.168.1.20:4747/video).
    """
    from urllib.parse import urlparse

    if fonte == "webcam":
        fonte = "0"
    if fonte.isdigit():
        return abrirCamera(int(fonte)), True
    if fonte.startswith(("http://", "https://")) or fonte.split(":")[0].count(".") == 3:
        url = fonte if fonte.startswith("http") else f"http://{fonte}"
        if urlparse(url).path in ("", "/"):   # so o IP (ESP32): o stream dela fica em /stream
            url = url.rstrip("/") + "/stream"
        return LeitorMJPEG(url), True
    return cv.VideoCapture(fonte), False
