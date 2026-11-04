"""
Desenhe no ar com gestos de mao, pela webcam ou pela ESP32-CAM.
Cada traco e um objeto separado: os gestos agem no desenho que esta sob a sua mao.

Gestos:
  pinca (polegar encostando no indicador)        -> desenha um traco novo
  rock (indicador + mindinho em pe) e gira a mao -> gira o desenho sob a mao
  pinca com as DUAS maos em volta de um desenho  -> move e redimensiona esse desenho
  mao fechada (punho)                            -> borracha
  tocar uma cor com a ponta do indicador         -> troca a cor
  pinca na barra ao lado das cores, subir/descer -> muda a espessura

Teclas:
  Z desfaz | C limpa | F tela cheia | H ajuda | S salva a imagem | Q ou ESC sai

Uso:
    pip install mediapipe
    python gestos/desenhar.py
    python gestos/desenhar.py --fonte 192.168.1.6
"""
import argparse
import math
import os
import sys
import time
import urllib.request

import cv2 as cv
import numpy as np

PASTA = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(PASTA)
MODELO_MAO = os.path.join(PASTA, "hand_landmarker.task")
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")
JANELA = "L.A.R.A | desenho com gestos"

sys.path.insert(0, os.path.join(RAIZ, "rf-detr"))
from comum import abrirFonte  # noqa: E402  (webcam, video ou ESP32)

# cores em BGR
CORES = [(255, 229, 0), (109, 42, 255), (0, 237, 204), (26, 122, 255), (255, 255, 255), (68, 68, 255)]
ESP_MIN, ESP_MAX = 2, 40

# pontos da mao no MediaPipe
PULSO, POLEGAR, INDICADOR, MEDIO, ANELAR, MINDINHO = 0, 4, 8, 12, 16, 20
ARTICULACAO = {INDICADOR: 6, MEDIO: 10, ANELAR: 14, MINDINHO: 18}
BASE_MEDIO = 9
PALMA = [0, 5, 9, 13, 17]

PINCA_FECHA = 0.30      # distancia polegar-indicador / tamanho da palma (so na imagem 2D)
PINCA_ABRE = 0.45       # histerese: precisa abrir mais pra soltar (evita piscar)
PINCA_FECHA_M = 0.040   # mesma coisa em 3D, em metros: nao muda quando a mao inclina
PINCA_ABRE_M = 0.065
PINCA_2D_REFORCO = 0.22 # na imagem 2D: tao junto que e pinca mesmo se o 3D errar (ex.: dedo escondido)
ESPERA_DESENHO = 0.12   # s de pinca antes de comecar a desenhar
DESCARTA_TRACO = 0.5    # s: traco mais curto que isso e descartado se virar gesto de 2 maos
ESPERA_COR = 0.25       # s com o dedo em cima da cor pra trocar
SUMIU = 0.15            # s sem ver a mao: tolera falhas curtas do detector


# ---------------------------------------------------------------- matrizes 2D

def translacao(v):
    m = np.eye(3)
    m[:2, 2] = v
    return m


def escala(s):
    return np.diag([s, s, 1.0])


def rotacao(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def aplicar(m, pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return pts @ m[:2, :2].T + m[:2, 2]


def angulo_limpo(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def dist_linha(p, pts):
    """Menor distancia de um ponto ate uma linha poligonal."""
    if len(pts) == 1:
        return float(np.linalg.norm(pts[0] - p))
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    return float(np.min(np.linalg.norm(a + ab * t[:, None] - p, axis=1)))


# ---------------------------------------------------------------- maos

def baixarModelo():
    if os.path.exists(MODELO_MAO):
        return
    print("baixando o modelo de maos do MediaPipe (uns 8 MB)...")
    urllib.request.urlretrieve(URL_MODELO, MODELO_MAO)


def criarDetector(num_maos=2):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opcoes = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELO_MAO),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_maos,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,      # mais tolerante: nao perde a mao tao facil quando ela borra
    )
    return mp, vision.HandLandmarker.create_from_options(opcoes)


_proxima_mao = iter(range(1, 10 ** 9))


def associarMaos(resultado, maos, w, h, agora):
    """
    Atualiza o dicionario maos (chave -> Mao) com as maos detectadas neste frame e devolve as chaves.
    Cada deteccao vai pra Mao conhecida mais PERTO, e nao pelo rotulo esquerda/direita do MediaPipe,
    que as vezes chama as duas de 'direita' e faria uma mao perder o estado (ex.: a pinca) a cada frame.
    """
    mundos = getattr(resultado, "hand_world_landmarks", None) or []
    chaves = []
    for i, marcos in enumerate(resultado.hand_landmarks):
        pts = np.array([[p.x * w, p.y * h] for p in marcos])
        mundo = np.array([[p.x, p.y, p.z] for p in mundos[i]]) if i < len(mundos) else None
        centro = pts[PALMA].mean(axis=0)
        candidatas = [(np.linalg.norm(m.centro - centro), k) for k, m in maos.items()
                      if k not in chaves and hasattr(m, "centro") and agora - m.vista <= 1.0]
        dist, chave = min(candidatas) if candidatas else (1e9, None)
        if chave is None or dist > 0.35 * w:
            chave = f"mao{next(_proxima_mao)}"
        maos.setdefault(chave, Mao()).atualizar(pts, agora, mundo)
        chaves.append(chave)
    return chaves


class Mao:
    """Estado de uma mao entre frames: gesto atual, ponto da pinca suavizado etc."""

    def __init__(self):
        self.pinca = False
        self.inicio_pinca = 0.0
        self.ponto_inicio = None   # onde a pinca comecou (decide se e desenho ou barra)
        self.travada = False       # usada num gesto de 2 maos: so desenha depois de soltar
        self.ponto = None
        self.vista = 0.0

    def atualizar(self, pts, agora, mundo=None):
        """pts: 21 pontos na imagem (px). mundo: os mesmos 21 pontos em 3D (metros), se tiver."""
        dt = agora - self.vista
        if dt > SUMIU:   # a mao saiu da imagem e voltou: comeca do zero
            self.pinca, self.travada, self.ponto = False, False, None
        palma = np.linalg.norm(pts[BASE_MEDIO] - pts[PULSO]) + 1e-6
        self.palma = palma

        def esticado(dedo):
            return np.linalg.norm(pts[dedo] - pts[PULSO]) > 1.15 * np.linalg.norm(pts[ARTICULACAO[dedo]] - pts[PULSO])

        def dobrado(dedo):
            return np.linalg.norm(pts[dedo] - pts[PULSO]) < 1.0 * np.linalg.norm(pts[ARTICULACAO[dedo]] - pts[PULSO])

        # punho vem primeiro: com a mao fechada o polegar encosta no indicador e pareceria pinca
        self.punho = all(dobrado(d) for d in (INDICADOR, MEDIO, ANELAR, MINDINHO))

        antes = self.pinca
        dist2 = np.linalg.norm(pts[POLEGAR] - pts[INDICADOR]) / palma
        self.dist_cm = None
        if mundo is not None:
            # 3D: distancia real entre as pontas dos dedos, nao depende do angulo da mao;
            # o 2D entra de reforco quando os dedos estao claramente colados na imagem
            dist3 = np.linalg.norm(mundo[POLEGAR] - mundo[INDICADOR])
            self.dist_cm = dist3 * 100
            junto = dist3 < (PINCA_ABRE_M if antes else PINCA_FECHA_M) or \
                dist2 < (PINCA_2D_REFORCO * 1.4 if antes else PINCA_2D_REFORCO)   # solta rapido
        else:
            junto = dist2 < (PINCA_ABRE if antes else PINCA_FECHA)
        self.pinca = not self.punho and junto

        alvo = (pts[POLEGAR] + pts[INDICADOR]) / 2
        if self.ponto is None:
            self.ponto = alvo
        else:
            # suavizacao que se adapta: devagar = firme (sem tremer); rapido = sem atraso
            velocidade = np.linalg.norm(alvo - self.ponto) / max(dt, 1e-3)
            a = min(max(velocidade / 1500, 0.5), 0.95)
            self.ponto = a * alvo + (1 - a) * self.ponto
        if self.pinca and not antes:
            self.inicio_pinca = agora
            self.ponto_inicio = alvo.copy()   # sem suavizar: decide se e desenho ou barra
        if not self.pinca:
            self.travada = False

        self.rock = (not self.pinca and not self.punho and esticado(INDICADOR) and esticado(MINDINHO)
                     and not esticado(MEDIO) and not esticado(ANELAR))
        self.apontando = not self.pinca and not self.punho and not self.rock and esticado(INDICADOR)
        self.ponta = pts[INDICADOR]
        self.centro = pts[PALMA].mean(axis=0)
        v = pts[BASE_MEDIO] - pts[PULSO]
        self.angulo = math.atan2(v[1], v[0])
        self.vista = agora


# ---------------------------------------------------------------- desenhos

class Objeto:
    """Um traco: pontos em coordenadas proprias + a matriz M que leva pra tela."""

    def __init__(self, cor, esp, M=None):
        self.pts = []
        self.cor = cor
        self.esp = esp
        self.M = np.eye(3) if M is None else M.copy()

    def tela(self):
        return aplicar(self.M, self.pts)

    def fator(self):
        return math.sqrt(abs(np.linalg.det(self.M[:2, :2])))

    def centro(self):
        return self.tela().mean(axis=0)

    def adicionar(self, ponto_tela):
        if self.pts and np.linalg.norm(self.tela()[-1] - ponto_tela) < 2:
            return
        self.pts.append(aplicar(np.linalg.inv(self.M), ponto_tela)[0].tolist())


def selecionar(objetos, p):
    """O desenho sob o ponto: o menor cuja area contem o ponto, senao o traco mais perto."""
    dentro, perto, melhor = [], None, 1e9
    for o in objetos:
        t = o.tela()
        folga = 20 + o.esp * o.fator()
        lo, hi = t.min(axis=0) - folga, t.max(axis=0) + folga
        if (p >= lo).all() and (p <= hi).all():
            dentro.append((float(np.prod(hi - lo)), o))
        d = dist_linha(p, t)
        if d < melhor:
            melhor, perto = d, o
    if dentro:
        return min(dentro, key=lambda x: x[0])[1]
    return perto if melhor < 80 else None


def apagar(objetos, centro, raio):
    """Borracha: tira os pontos dentro do circulo e divide o traco no que sobrou."""
    novos = []
    for o in objetos:
        t = o.tela()
        if dist_linha(centro, t) > raio + o.esp * o.fator() / 2:   # nem encostou nesse traco
            novos.append(o)
            continue
        denso = [t[0]]
        for a, b in zip(t[:-1], t[1:]):
            n = int(np.linalg.norm(b - a) // 4)
            denso.extend(a + (b - a) * k / (n + 1) for k in range(1, n + 1))
            denso.append(b)
        denso = np.array(denso)
        fora = np.linalg.norm(denso - centro, axis=1) > raio + o.esp * o.fator() / 2
        if fora.all():
            novos.append(o)
            continue
        inversa = np.linalg.inv(o.M)
        i = 0
        while i < len(denso):
            if not fora[i]:
                i += 1
                continue
            j = i
            while j < len(denso) and fora[j]:
                j += 1
            if j - i >= 2:
                pedaco = Objeto(o.cor, o.esp, o.M)
                pedaco.pts = aplicar(inversa, denso[i:j]).tolist()
                novos.append(pedaco)
            i = j
    return novos


def desenharObjetos(frame, objetos, destaque=None):
    if not objetos:
        return frame
    brilho = np.zeros_like(frame)
    linhas = []
    for o in objetos:
        if len(o.pts) < 2:
            continue
        pts = o.tela().round().astype(np.int32)
        esp = max(1, int(round(o.esp * o.fator())))
        linhas.append((o, pts, esp))
        cv.polylines(brilho, [pts], False, o.cor, esp * 3, cv.LINE_AA)
    brilho = cv.GaussianBlur(brilho, (0, 0), 9)
    saida = cv.add(frame, brilho)
    for o, pts, esp in linhas:
        if o is destaque:
            cv.polylines(saida, [pts], False, (255, 255, 255), esp + 6, cv.LINE_AA)
        cv.polylines(saida, [pts], False, o.cor, esp, cv.LINE_AA)
        cv.polylines(saida, [pts], False, (255, 255, 255), max(1, esp // 3), cv.LINE_AA)
    return saida


# ---------------------------------------------------------------- paleta e barra

class Painel:
    """Paleta de cores + barra de espessura, na lateral esquerda."""

    def __init__(self, h):
        self.raio = 24
        passo = min(68, (h - 200) / len(CORES))
        self.topo = 100
        self.cores = [(52, int(self.topo + passo * i + self.raio)) for i in range(len(CORES))]
        self.barra_x = 122
        self.barra_y0 = self.topo + self.raio // 2
        self.barra_y1 = int(self.cores[-1][1] + self.raio // 2)
        self.largura = 160   # nao desenha nessa faixa

    def corEm(self, p):
        for i, c in enumerate(self.cores):
            if np.linalg.norm(np.array(c) - p) <= self.raio + 8:
                return i
        return None

    def naBarra(self, p):
        return abs(p[0] - self.barra_x) < 34 and self.barra_y0 - 24 <= p[1] <= self.barra_y1 + 24

    def espessuraEm(self, y):
        f = (self.barra_y1 - y) / (self.barra_y1 - self.barra_y0)
        return int(round(ESP_MIN + min(max(f, 0), 1) * (ESP_MAX - ESP_MIN)))

    def desenhar(self, img, cor_idx, espessura, mirando=None, progresso=0.0, ajustando=False):
        fundo = img.copy()
        cv.rectangle(fundo, (12, self.topo - 30), (self.largura - 8, self.barra_y1 + 52), (20, 20, 20), -1)
        cv.addWeighted(fundo, 0.45, img, 0.55, 0, img)
        for i, (c, centro) in enumerate(zip(CORES, self.cores)):
            cv.circle(img, centro, self.raio, c, -1, cv.LINE_AA)
            if i == cor_idx:
                cv.circle(img, centro, self.raio + 6, (255, 255, 255), 3, cv.LINE_AA)
            if i == mirando and progresso > 0:
                cv.ellipse(img, centro, (self.raio + 11, self.raio + 11), -90, 0, 360 * progresso,
                           (255, 255, 255), 2, cv.LINE_AA)

        x, y0, y1 = self.barra_x, self.barra_y0, self.barra_y1
        cv.line(img, (x, y0), (x, y1), (90, 90, 90), 6, cv.LINE_AA)
        y = int(y1 - (espessura - ESP_MIN) / (ESP_MAX - ESP_MIN) * (y1 - y0))
        cv.line(img, (x, y), (x, y1), CORES[cor_idx], 6, cv.LINE_AA)
        cv.circle(img, (x, y), 13 if ajustando else 10, (255, 255, 255), -1, cv.LINE_AA)
        # previa da espessura
        cv.circle(img, (int((12 + self.largura - 8) / 2), y1 + 30), max(1, espessura // 2), CORES[cor_idx], -1, cv.LINE_AA)
        cv.putText(img, str(espessura), (x - 12, y0 - 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)


# ---------------------------------------------------------------- tela

def texto(img, msg, pos, tamanho=0.7, cor=(255, 255, 255), grossura=2):
    cv.putText(img, msg, pos, cv.FONT_HERSHEY_SIMPLEX, tamanho, (0, 0, 0), grossura + 3, cv.LINE_AA)
    cv.putText(img, msg, pos, cv.FONT_HERSHEY_SIMPLEX, tamanho, cor, grossura, cv.LINE_AA)


AJUDA = [
    "pinca: desenha  |  punho: apaga",
    "rock + girar a mao: gira o desenho embaixo da mao",
    "pinca com as 2 maos em volta de um desenho: move e redimensiona",
    "indicador na cor: troca a cor  |  pinca na barra: espessura",
    "Z desfaz  C limpa  F tela cheia  S salva  H ajuda  Q sai",
]


def main():
    parser = argparse.ArgumentParser(description="Desenhe no ar com gestos de mao.")
    parser.add_argument("--fonte", default="webcam", help="webcam, numero da camera, video ou IP da ESP32")
    parser.add_argument("--largura", type=int, default=1280, help="largura da tela em pixels")
    parser.add_argument("--altura", type=int, default=720)
    parser.add_argument("--cheia", action="store_true", help="comeca em tela cheia")
    parser.add_argument("--sem-espelho", action="store_true", help="nao espelha a imagem")
    args = parser.parse_args()

    baixarModelo()
    mp, detector = criarDetector()

    try:
        captura, _ = abrirFonte(args.fonte)
    except OSError as e:
        raise SystemExit(f"nao conectou na fonte {args.fonte}: {e}")
    if isinstance(captura, cv.VideoCapture):
        captura.set(cv.CAP_PROP_FRAME_WIDTH, args.largura)
        captura.set(cv.CAP_PROP_FRAME_HEIGHT, args.altura)

    cv.namedWindow(JANELA, cv.WINDOW_NORMAL)
    cv.resizeWindow(JANELA, args.largura, args.altura)
    cheia = args.cheia
    if cheia:
        cv.setWindowProperty(JANELA, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    objetos = []            # cada traco e um objeto
    atual = None            # traco sendo desenhado
    inicio_atual = 0.0
    maos = {}
    modo = "livre"
    gesto = {}
    cor_idx, espessura = 0, 8
    mira_cor = {"idx": None, "desde": 0.0}
    ajuda = True
    painel = None
    ultimo_ts = -1
    fps = 0.0

    def terminarTraco(descartar=False):
        nonlocal atual
        if atual is not None and len(atual.pts) >= 2 and not descartar:
            objetos.append(atual)
        atual = None

    try:
        while True:
            t0 = time.time()
            ok, frame = captura.read()
            if not ok:
                break
            if not args.sem_espelho:
                frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]
            if w != args.largura:   # deixa tudo no tamanho da tela (a ESP32 manda 640x480)
                frame = cv.resize(frame, (args.largura, round(h * args.largura / w)))
                h, w = frame.shape[:2]
            if painel is None:
                painel = Painel(h)

            # ---- maos
            rgb = np.ascontiguousarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            ts = max(int(time.monotonic() * 1000), ultimo_ts + 1)
            ultimo_ts = ts
            resultado = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            agora = time.time()
            associarMaos(resultado, maos, w, h, agora)
            for chave in [k for k, m in maos.items() if agora - m.vista > 1.0]:
                del maos[chave]
            # maos vistas agora ou ha pouquinho (o detector as vezes perde a mao por 1-2 frames)
            ativas = [m for m in maos.values() if agora - m.vista <= SUMIU]

            punho = next((m for m in ativas if m.punho), None)
            pincando = [m for m in ativas if m.pinca]
            rock = next((m for m in ativas if m.rock), None)
            destaque = None
            mirando, progresso = None, 0.0

            # ---- gestos (punho > 2 pincas > rock > 1 pinca > apontar pra cor)
            if punho is not None:
                if modo != "apagando":
                    terminarTraco()
                    modo = "apagando"
                raio = max(28.0, 0.55 * punho.palma)
                gesto = {"centro": punho.centro, "raio": raio}
                objetos[:] = apagar(objetos, punho.centro, raio)

            elif len(pincando) >= 2:
                a, b = pincando[0], pincando[1]
                if modo != "transformando":
                    terminarTraco(descartar=agora - inicio_atual < DESCARTA_TRACO)
                    modo = "transformando"
                    meio0 = (a.ponto + b.ponto) / 2
                    alvo = selecionar(objetos, meio0) or selecionar(objetos, a.ponto) or selecionar(objetos, b.ponto)
                    gesto = {"alvo": alvo, "M0": alvo.M.copy() if alvo else None, "meio0": meio0,
                             "dist0": max(np.linalg.norm(a.ponto - b.ponto), 1.0)}
                a.travada = b.travada = True
                alvo = gesto["alvo"]
                if alvo in objetos:
                    meio = (a.ponto + b.ponto) / 2
                    s = min(max(np.linalg.norm(a.ponto - b.ponto) / gesto["dist0"], 0.1), 10)
                    alvo.M = translacao(meio) @ escala(s) @ translacao(-gesto["meio0"]) @ gesto["M0"]
                    destaque = alvo

            elif rock is not None:
                if modo != "girando":
                    terminarTraco()
                    modo = "girando"
                    alvo = selecionar(objetos, rock.centro) or selecionar(objetos, rock.ponta)
                    gesto = {"alvo": alvo, "M0": alvo.M.copy() if alvo else None, "ultimo": rock.angulo,
                             "total": 0.0, "centro": alvo.centro() if alvo else rock.centro}
                gesto["total"] += angulo_limpo(rock.angulo - gesto["ultimo"])
                gesto["ultimo"] = rock.angulo
                alvo = gesto["alvo"]
                if alvo in objetos:
                    c = gesto["centro"]
                    alvo.M = translacao(c) @ rotacao(gesto["total"]) @ translacao(-c) @ gesto["M0"]
                    destaque = alvo

            elif len(pincando) == 1 and not pincando[0].travada:
                m = pincando[0]
                if painel.naBarra(m.ponto_inicio):
                    modo = "espessura"
                    espessura = painel.espessuraEm(m.ponto[1])
                elif m.ponto_inicio[0] < painel.largura:
                    modo = "livre"        # pinca que comecou em cima da paleta: ignora
                elif agora - m.inicio_pinca >= ESPERA_DESENHO:
                    if modo != "desenhando":
                        modo = "desenhando"
                        atual = Objeto(CORES[cor_idx], espessura)
                        inicio_atual = agora
                    atual.adicionar(m.ponto)

            else:
                if modo == "desenhando":
                    terminarTraco()
                modo = "livre"

            # apontar pra uma cor (com qualquer mao que nao esteja fazendo gesto)
            for m in ativas:
                if m.apontando:
                    i = painel.corEm(m.ponta)
                    if i is not None:
                        mirando = i
                        break
            if mirando is None or mirando != mira_cor["idx"]:
                mira_cor = {"idx": mirando, "desde": agora}
            elif mirando is not None:
                progresso = min((agora - mira_cor["desde"]) / ESPERA_COR, 1.0)
                if progresso >= 1.0:
                    cor_idx = mirando

            # ---- tela
            lista = objetos + ([atual] if atual is not None and len(atual.pts) >= 2 else [])
            saida = desenharObjetos(frame, lista, destaque)
            painel.desenhar(saida, cor_idx, espessura, mirando, progresso, modo == "espessura")
            cor = CORES[cor_idx]

            for m in ativas:
                if m.punho:
                    continue
                if m.apontando and m.ponta[0] < painel.largura + 20:
                    cv.circle(saida, tuple(int(v) for v in m.ponta), 7, (255, 255, 255), -1, cv.LINE_AA)
                p = tuple(int(v) for v in m.ponto)
                if m.pinca:
                    cv.circle(saida, p, max(4, espessura // 2 + 3), cor, -1, cv.LINE_AA)
                else:
                    cv.circle(saida, p, 10, cor, 2, cv.LINE_AA)
            if modo == "apagando":
                c = tuple(int(v) for v in gesto["centro"])
                cv.circle(saida, c, int(gesto["raio"]), (255, 255, 255), 2, cv.LINE_AA)
                cv.circle(saida, c, int(gesto["raio"]), (0, 0, 0), 1, cv.LINE_AA)
            if modo == "girando" and gesto.get("alvo") in objetos:
                c = tuple(int(v) for v in gesto["centro"])
                cv.circle(saida, c, 6, (255, 255, 255), -1, cv.LINE_AA)
                cv.ellipse(saida, c, (40, 40), 0, 0, math.degrees(gesto["total"]) % 360 or 1, cor, 3, cv.LINE_AA)
            if modo == "transformando" and len(pincando) >= 2:
                cv.line(saida, tuple(int(v) for v in pincando[0].ponto), tuple(int(v) for v in pincando[1].ponto),
                        (255, 255, 255), 1, cv.LINE_AA)

            gasto = max(time.time() - t0, 1e-3)
            fps = 0.9 * fps + 0.1 / gasto if fps else 1 / gasto
            nomes_modo = {"livre": "", "desenhando": "desenhando", "girando": "girando",
                          "transformando": "movendo / redimensionando", "apagando": "apagando",
                          "espessura": f"espessura {espessura}"}
            if modo in ("girando", "transformando") and gesto.get("alvo") is None:
                nomes_modo[modo] = "nenhum desenho embaixo da mao"
            texto(saida, f"{fps:.0f} FPS  |  {len(objetos)} desenho(s)", (painel.largura + 10, 34), 0.6)
            if nomes_modo[modo]:
                texto(saida, nomes_modo[modo], (painel.largura + 10, 70), 0.9, cor)
            if ajuda:
                for i, linha in enumerate(AJUDA):
                    texto(saida, linha, (painel.largura + 10, h - 20 - (len(AJUDA) - 1 - i) * 28), 0.55)
            cv.imshow(JANELA, saida)

            # ---- teclado
            tecla = cv.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if cv.getWindowProperty(JANELA, cv.WND_PROP_VISIBLE) < 1:
                break
            if tecla == ord("z") and objetos:
                objetos.pop()
            elif tecla == ord("c"):
                objetos.clear()
                atual = None
                modo = "livre"
            elif tecla == ord("h"):
                ajuda = not ajuda
            elif tecla == ord("f"):
                cheia = not cheia
                cv.setWindowProperty(JANELA, cv.WND_PROP_FULLSCREEN,
                                     cv.WINDOW_FULLSCREEN if cheia else cv.WINDOW_NORMAL)
            elif tecla == ord("s"):
                pasta = os.path.join(PASTA, "capturas")
                os.makedirs(pasta, exist_ok=True)
                caminho = os.path.join(pasta, time.strftime("desenho_%Y%m%d_%H%M%S.png"))
                cv.imwrite(caminho, saida)
                print(f"salvo: {caminho}")
    finally:
        captura.release()
        detector.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
