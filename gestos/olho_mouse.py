"""
Mouse controlado pelo OLHAR (webcam), com os botoes nos gestos da mao.

  olhar                                  -> move o cursor
  pinca com a mao                        -> botao ESQUERDO (segurar = arrastar)
  paz (indicador + medio em V)           -> clique DIREITO
  punho fechado                          -> congela o cursor (pra olhar pra outro lugar sem mexer)

Na primeira vez ele abre uma CALIBRACAO em tela cheia: olhe para cada ponto ate ele sumir.
A calibracao fica salva em gestos/calibracao_olho.json.

Teclas (com a janelinha da camera em foco):
  R recalibra | O recentraliza (1 ponto, rapido) | P liga/desliga | Q ou ESC sai

Dicas: luz no rosto (de frente, nao atras de voce), camera na altura dos olhos,
cabeca mais ou menos parada. Webcam nao e eye tracker de laboratorio: a precisao fica
na faixa de um botao grande / icone da area de trabalho.

Uso:
    python gestos/olho_mouse.py
    python gestos/olho_mouse.py --calibrar        # forca uma calibracao nova
"""
import argparse
import json
import os
import time
import urllib.request
from collections import deque

import cv2 as cv
import numpy as np

from desenhar import PASTA, abrirFonte, baixarModelo, criarDetector
from mouse import CONEXOES, CONFIRMA, Mouse, classificar

MODELO_ROSTO = os.path.join(PASTA, "face_landmarker.task")
URL_ROSTO = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")
CALIBRACAO = os.path.join(PASTA, "calibracao_olho.json")
JANELA = "L.A.R.A | mouse pelo olhar"
JANELA_CAL = "L.A.R.A | calibracao"
TRAVA_CLIQUE = 0.35

# pontos do rosto (MediaPipe Face Mesh, 478 pontos com a iris)
OLHO_A = {"cantos": (33, 133), "cima": 159, "baixo": 145, "iris": 468}
OLHO_B = {"cantos": (362, 263), "cima": 386, "baixo": 374, "iris": 473}
NARIZ = 1

# ordem padrao dos blendshapes (usada se vierem sem nome)
BLENDSHAPES = {"eyeBlinkLeft": 9, "eyeBlinkRight": 10, "eyeLookDownLeft": 11, "eyeLookDownRight": 12,
               "eyeLookInLeft": 13, "eyeLookInRight": 14, "eyeLookOutLeft": 15, "eyeLookOutRight": 16,
               "eyeLookUpLeft": 17, "eyeLookUpRight": 18}


# ---------------------------------------------------------------- rosto -> caracteristicas do olhar

def criarDetectorRosto():
    if not os.path.exists(MODELO_ROSTO):
        print("baixando o modelo de rosto do MediaPipe (uns 4 MB)...")
        urllib.request.urlretrieve(URL_ROSTO, MODELO_ROSTO)
    from mediapipe.tasks.python import BaseOptions, vision

    opcoes = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELO_ROSTO),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opcoes)


def posicaoIris(pts, olho):
    """Onde a iris esta dentro do olho: x de 0 (canto da esquerda) a 1 (direita), y relativo a abertura."""
    a, b = pts[olho["cantos"][0]], pts[olho["cantos"][1]]
    if a[0] > b[0]:
        a, b = b, a
    eixo = b - a
    largura = np.linalg.norm(eixo) + 1e-6
    u = eixo / largura
    n = np.array([-u[1], u[0]])
    iris = pts[olho["iris"]]
    meio_vertical = (pts[olho["cima"]] + pts[olho["baixo"]]) / 2
    return np.dot(iris - a, u) / largura, np.dot(iris - meio_vertical, n) / largura


def caracteristicas(resultado, w, h):
    """Vetor que descreve pra onde a pessoa olha, ou None (sem rosto ou piscando)."""
    if not resultado.face_landmarks:
        return None, None
    pts = np.array([[p.x * w, p.y * h] for p in resultado.face_landmarks[0]])
    if len(pts) < 478:
        raise SystemExit("o modelo de rosto nao devolveu os pontos da iris")

    bs = {}
    if resultado.face_blendshapes:
        categorias = resultado.face_blendshapes[0]
        for nome, i in BLENDSHAPES.items():
            achou = next((c.score for c in categorias if c.category_name == nome), None)
            bs[nome] = achou if achou is not None else (categorias[i].score if i < len(categorias) else 0.0)
    if bs and max(bs.get("eyeBlinkLeft", 0), bs.get("eyeBlinkRight", 0)) > 0.45:
        return None, pts   # piscando: os olhos fechados dao leitura errada

    xa, ya = posicaoIris(pts, OLHO_A)
    xb, yb = posicaoIris(pts, OLHO_B)
    olhar_h = (bs.get("eyeLookOutLeft", 0) + bs.get("eyeLookInRight", 0)
               - bs.get("eyeLookInLeft", 0) - bs.get("eyeLookOutRight", 0)) / 2
    olhar_v = (bs.get("eyeLookUpLeft", 0) + bs.get("eyeLookUpRight", 0)
               - bs.get("eyeLookDownLeft", 0) - bs.get("eyeLookDownRight", 0)) / 2

    yaw = pitch = 0.0
    if resultado.facial_transformation_matrixes:
        R = np.asarray(resultado.facial_transformation_matrixes[0])[:3, :3]
        yaw, pitch = R[0, 2], R[1, 2]   # pra onde a cabeca aponta
    nariz = pts[NARIZ] / [w, h]
    return np.array([(xa + xb) / 2, (ya + yb) / 2, olhar_h, olhar_v, yaw, pitch, nariz[0], nariz[1]]), pts


# ---------------------------------------------------------------- regressao (caracteristicas -> tela)

def polinomio(X):
    """Termos de grau 2: 1, x_i, x_i*x_j."""
    X = np.atleast_2d(X)
    n = X.shape[1]
    termos = [np.ones(len(X))] + [X[:, i] for i in range(n)]
    termos += [X[:, i] * X[:, j] for i in range(n) for j in range(i, n)]
    return np.stack(termos, axis=1)


class Modelo:
    def __init__(self, dados=None):
        self.dados = dados

    def treinar(self, X, Y, tela, lambda_=2.0):
        X, Y = np.asarray(X), np.asarray(Y)
        media, desvio = X.mean(axis=0), X.std(axis=0) + 1e-6
        P = polinomio((X - media) / desvio)
        reg = lambda_ * np.eye(P.shape[1])
        reg[0, 0] = 0
        W = np.linalg.solve(P.T @ P + reg, P.T @ Y)
        self.dados = {"media": media.tolist(), "desvio": desvio.tolist(), "W": W.tolist(),
                      "tela": list(tela), "deslocamento": [0.0, 0.0]}
        erro = np.linalg.norm(self.prever_varios(X) - Y, axis=1)
        return float(np.median(erro))

    def prever_varios(self, X):
        d = self.dados
        P = polinomio((np.asarray(X) - d["media"]) / np.asarray(d["desvio"]))
        return P @ np.asarray(d["W"]) + d["deslocamento"]

    def prever(self, f):
        return self.prever_varios(f[None])[0]

    def salvar(self):
        with open(CALIBRACAO, "w", encoding="utf-8") as arq:
            json.dump(self.dados, arq)

    @staticmethod
    def carregar(tela):
        if not os.path.exists(CALIBRACAO):
            return None
        with open(CALIBRACAO, encoding="utf-8") as arq:
            dados = json.load(arq)
        return Modelo(dados) if dados.get("tela") == list(tela) else None   # trocou a resolucao: recalibra


class Filtro:
    """Mediana dos ultimos frames (tira os pulos) + suavizacao com zona morta (tira o tremido)."""

    def __init__(self):
        self.ultimos = deque(maxlen=6)
        self.valor = None

    def __call__(self, p):
        self.ultimos.append(p)
        alvo = np.median(np.array(self.ultimos), axis=0)
        if self.valor is None:
            self.valor = alvo
            return self.valor
        dist = np.linalg.norm(alvo - self.valor)
        if dist < 22:
            return self.valor                       # olhar parado: cursor parado
        alfa = 0.45 if dist > 220 else 0.12         # mudou de lugar: vai rapido; ajuste fino: devagar
        self.valor = self.valor + alfa * (alvo - self.valor)
        return self.valor


# ---------------------------------------------------------------- calibracao

def pontosCalibracao(completa=True):
    if not completa:
        return [(0.5, 0.5)]
    grade = [(x, y) for y in (0.08, 0.5, 0.92) for x in (0.06, 0.5, 0.94)]
    extras = [(0.28, 0.29), (0.72, 0.29), (0.28, 0.71), (0.72, 0.71), (0.5, 0.5)]
    return grade + extras


def calibrar(captura, rosto, mp, mouse, espelhar, relogio, completa=True):
    """Mostra pontos em tela cheia e junta (caracteristicas, posicao na tela) enquanto voce olha."""
    W, H = mouse.largura, mouse.altura
    cv.namedWindow(JANELA_CAL, cv.WINDOW_NORMAL)
    cv.setWindowProperty(JANELA_CAL, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    fundo = np.full((H, W, 3), 18, np.uint8)

    def mensagem(img, linhas):
        for i, linha in enumerate(linhas):
            tam = cv.getTextSize(linha, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv.putText(img, linha, ((W - tam[0]) // 2, H // 2 - 60 + i * 50), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                       (230, 230, 230), 2, cv.LINE_AA)

    # espera o espaco
    while True:
        tela = fundo.copy()
        ok, frame = captura.read()
        if not ok:
            return None
        if espelhar:
            frame = cv.flip(frame, 1)
        f, _ = caracteristicas(detectar(rosto, mp, frame, relogio), frame.shape[1], frame.shape[0])
        status = "rosto encontrado" if f is not None else "procurando seu rosto... (luz de frente ajuda)"
        mensagem(tela, ["Calibracao do olhar" if completa else "Recentralizar",
                        "Olhe para cada ponto ate ele sumir. Mexa so os olhos.",
                        status, "ESPACO comeca  |  ESC cancela"])
        cv.imshow(JANELA_CAL, tela)
        tecla = cv.waitKey(1) & 0xFF
        if tecla == 27:
            cv.destroyWindow(JANELA_CAL)
            return None
        if tecla == 32 and f is not None:
            break

    X, Y = [], []
    for px, py in pontosCalibracao(completa):
        alvo = np.array([px * W, py * H])
        inicio = time.time()
        duracao = 2.2
        while time.time() - inicio < duracao:
            t = time.time() - inicio
            ok, frame = captura.read()
            if not ok:
                return None
            if espelhar:
                frame = cv.flip(frame, 1)
            f, _ = caracteristicas(detectar(rosto, mp, frame, relogio), frame.shape[1], frame.shape[0])
            if f is not None and t > 0.7:         # os primeiros 0.7 s sao o olho chegando no ponto
                X.append(f)
                Y.append(alvo)
            tela = fundo.copy()
            raio = int(30 - 22 * min(t / duracao, 1))
            centro = tuple(alvo.astype(int))
            cv.circle(tela, centro, raio + 8, (255, 229, 0), 2, cv.LINE_AA)
            cv.circle(tela, centro, max(raio // 3, 4), (255, 255, 255), -1, cv.LINE_AA)
            cv.imshow(JANELA_CAL, tela)
            if (cv.waitKey(1) & 0xFF) == 27:
                cv.destroyWindow(JANELA_CAL)
                return None
    cv.destroyWindow(JANELA_CAL)
    return np.array(X), np.array(Y)


# ---------------------------------------------------------------- deteccao

class Relogio:
    """Timestamps crescentes pro modo VIDEO do MediaPipe (um por detector)."""

    def __init__(self):
        self.ultimo = {}

    def __call__(self, chave):
        ts = max(int(time.monotonic() * 1000), self.ultimo.get(chave, -1) + 1)
        self.ultimo[chave] = ts
        return ts


def detectar(detector, mp, frame, relogio):
    rgb = np.ascontiguousarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    return detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), relogio(id(detector)))


# ---------------------------------------------------------------- principal

def main():
    parser = argparse.ArgumentParser(description="Mouse pelo olhar, botoes pelos gestos.")
    parser.add_argument("--fonte", default="webcam", help="webcam ou numero da camera (precisa ver seu rosto de perto)")
    parser.add_argument("--calibrar", action="store_true", help="faz uma calibracao nova")
    parser.add_argument("--sem-espelho", action="store_true")
    args = parser.parse_args()
    espelhar = not args.sem_espelho

    mouse = Mouse()
    baixarModelo()
    mp, maos = criarDetector(num_maos=1)
    rosto = criarDetectorRosto()
    relogio = Relogio()
    try:
        captura, _ = abrirFonte(args.fonte)
    except OSError as e:
        raise SystemExit(f"nao conectou na fonte {args.fonte}: {e}")
    if isinstance(captura, cv.VideoCapture):
        captura.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        captura.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    def calibracaoCompleta():
        dados = calibrar(captura, rosto, mp, mouse, espelhar, relogio, completa=True)
        if dados is None or len(dados[0]) < 60:
            print("calibracao cancelada ou com poucas leituras (o rosto sumiu?)")
            return None
        modelo = Modelo()
        erro = modelo.treinar(dados[0], dados[1], (mouse.largura, mouse.altura))
        modelo.salvar()
        print(f"calibrado com {len(dados[0])} leituras; erro tipico {erro:.0f} px")
        return modelo

    modelo = None if args.calibrar else Modelo.carregar((mouse.largura, mouse.altura))
    if modelo is None:
        modelo = calibracaoCompleta()
        if modelo is None:
            raise SystemExit("sem calibracao nao da pra usar o olhar")

    cv.namedWindow(JANELA, cv.WINDOW_NORMAL)
    cv.resizeWindow(JANELA, 480, 300)
    try:
        cv.setWindowProperty(JANELA, cv.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    filtro = Filtro()
    ligado = True
    gesto_visto, repeticoes, gesto = "mover", 0, "mover"
    esquerdo_apertado = False
    trava_ate = 0.0
    fps = 0.0
    print("R recalibra | O recentraliza | P liga/desliga | Q sai")

    try:
        while True:
            t0 = time.time()
            ok, frame = captura.read()
            if not ok:
                break
            if espelhar:
                frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]
            agora = time.time()

            # ---- olhar -> cursor
            f, pts_rosto = caracteristicas(detectar(rosto, mp, frame, relogio), w, h)
            if f is not None:
                alvo = filtro(modelo.prever(f))
                if ligado and gesto != "punho" and agora >= trava_ate:
                    mouse.mover(*alvo)

            # ---- mao -> botoes
            res_mao = detectar(maos, mp, frame, relogio)
            if res_mao.hand_landmarks:
                pts = np.array([[p.x * w, p.y * h] for p in res_mao.hand_landmarks[0]])
                atual = classificar(pts, gesto)
                repeticoes = repeticoes + 1 if atual == gesto_visto else 1
                gesto_visto = atual
                novo = gesto_visto if repeticoes >= CONFIRMA else gesto
                if ligado and novo != gesto:
                    if novo == "esquerdo" and not esquerdo_apertado:
                        mouse.apertar("esquerdo")
                        esquerdo_apertado = True
                        trava_ate = agora + TRAVA_CLIQUE
                    elif gesto == "esquerdo" and esquerdo_apertado:
                        mouse.soltar("esquerdo")
                        esquerdo_apertado = False
                    if novo == "direito":
                        mouse.clicar("direito")
                        trava_ate = agora + TRAVA_CLIQUE
                gesto = novo
                for a, b in CONEXOES:
                    cv.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (255, 229, 0), 2, cv.LINE_AA)
            else:
                gesto_visto, repeticoes, gesto = "mover", 0, "mover"
                if esquerdo_apertado:
                    mouse.soltar("esquerdo")
                    esquerdo_apertado = False

            # ---- janelinha
            if pts_rosto is not None:
                for olho in (OLHO_A, OLHO_B):
                    cv.circle(frame, tuple(pts_rosto[olho["iris"]].astype(int)), 4, (0, 237, 204), -1, cv.LINE_AA)
                    for i in olho["cantos"]:
                        cv.circle(frame, tuple(pts_rosto[i].astype(int)), 2, (255, 255, 255), -1, cv.LINE_AA)
            gasto = max(time.time() - t0, 1e-3)
            fps = 0.9 * fps + 0.1 / gasto if fps else 1 / gasto
            if not ligado:
                rotulo, cor = "DESLIGADO (P liga)", (255, 255, 255)
            elif f is None:
                rotulo, cor = ("piscando" if pts_rosto is not None else "sem rosto"), (150, 150, 150)
            else:
                rotulo = {"esquerdo": "ESQUERDO (segurando)", "direito": "DIREITO", "punho": "cursor congelado",
                          "mover": "seguindo o olhar"}[gesto]
                cor = {"esquerdo": (0, 237, 204), "direito": (109, 42, 255)}.get(gesto, (255, 255, 255))
            for esp, c in ((5, (0, 0, 0)), (2, cor)):
                cv.putText(frame, rotulo, (14, 40), cv.FONT_HERSHEY_SIMPLEX, 1.1, c, esp, cv.LINE_AA)
            cv.putText(frame, f"{fps:.0f} FPS  |  R recalibra  O recentraliza  P pausa  Q sai", (14, h - 16),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
            cv.imshow(JANELA, frame)

            # ---- teclado
            tecla = cv.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27) or cv.getWindowProperty(JANELA, cv.WND_PROP_VISIBLE) < 1:
                break
            if tecla == ord("p"):
                ligado = not ligado
                if not ligado and esquerdo_apertado:
                    mouse.soltar("esquerdo")
                    esquerdo_apertado = False
            elif tecla == ord("r"):
                novo_modelo = calibracaoCompleta()
                if novo_modelo is not None:
                    modelo, filtro = novo_modelo, Filtro()
            elif tecla == ord("o"):
                dados = calibrar(captura, rosto, mp, mouse, espelhar, relogio, completa=False)
                if dados is not None and len(dados[0]) >= 10:
                    erro = np.median(dados[1] - modelo.prever_varios(dados[0]), axis=0)
                    modelo.dados["deslocamento"] = (np.asarray(modelo.dados["deslocamento"]) + erro).tolist()
                    modelo.salvar()
                    filtro = Filtro()
                    print(f"recentralizado ({erro[0]:+.0f}, {erro[1]:+.0f}) px")
    finally:
        if esquerdo_apertado:
            mouse.soltar("esquerdo")
        captura.release()
        maos.close()
        rosto.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
