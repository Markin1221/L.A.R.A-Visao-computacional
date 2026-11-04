"""
Mouse sem as maos: a CABECA move o cursor e o ROSTO faz os botoes.

  virar a cabeca                 -> move o cursor (pra onde o nariz aponta)
  sorrir                         -> botao ESQUERDO (manter o sorriso = segurar/arrastar)
  boca em "O" (tipo "ooh")       -> clique DIREITO

Na primeira vez abre uma calibracao rapida (5 pontos): aponte o NARIZ pra cada ponto,
com o rosto neutro (ela tambem aprende como e a sua boca parada).
Fica salva em gestos/calibracao_cabeca.json.

Teclas (com a janelinha da camera em foco):
  + / - sensibilidade | R recalibra | O recentraliza (olhe pro meio da tela e aperte)
  P liga/desliga | Q ou ESC sai

Uso:
    python gestos/cabeca_mouse.py
    python gestos/cabeca_mouse.py --calibrar
    python gestos/cabeca_mouse.py --sorriso 0.3      # precisa sorrir menos pra clicar
"""
import argparse
import json
import math
import os
import time

import cv2 as cv
import numpy as np

from desenhar import PASTA, abrirFonte
from mouse import CONFIRMA, Mouse
from olho_mouse import JANELA_CAL, Relogio, criarDetectorRosto, detectar

CALIBRACAO = os.path.join(PASTA, "calibracao_cabeca.json")
JANELA = "L.A.R.A | mouse pela cabeca"
TRAVA_CLIQUE = 0.4      # s com o cursor parado no clique (sorrir/abrir a boca mexe um pouco a cabeca)
NARIZ, TESTA, QUEIXO, BOCHECHA_E, BOCHECHA_D = 1, 10, 152, 234, 454

# onde ficam os pontos da calibracao (fracao da tela)
PONTOS = [("centro", 0.5, 0.5), ("esquerda", 0.05, 0.5), ("direita", 0.95, 0.5),
          ("cima", 0.5, 0.06), ("baixo", 0.5, 0.94)]

# expressoes do MediaPipe (nome, indice padrao se vierem sem nome)
BLENDSHAPES = {"jawOpen": 25, "mouthFunnel": 32, "mouthPucker": 38, "mouthSmileLeft": 44, "mouthSmileRight": 45}
NEUTRO_PADRAO = {"sorriso": 0.05, "o": 0.0}


# ---------------------------------------------------------------- rosto

def lerRosto(resultado, w, h):
    """(yaw, pitch) em graus, pontos do rosto e as expressoes {'sorriso', 'o'} (0 a 1)."""
    if not resultado.face_landmarks:
        return None, None, None
    pts = np.array([[p.x * w, p.y * h] for p in resultado.face_landmarks[0]])

    if resultado.facial_transformation_matrixes:
        R = np.asarray(resultado.facial_transformation_matrixes[0])[:3, :3]
        yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
        pitch = math.degrees(math.atan2(-R[1, 2], math.hypot(R[0, 2], R[2, 2])))
    else:
        # sem a matriz: posicao do nariz entre as bochechas / entre testa e queixo
        largura = np.linalg.norm(pts[BOCHECHA_D] - pts[BOCHECHA_E]) + 1e-6
        altura = np.linalg.norm(pts[QUEIXO] - pts[TESTA]) + 1e-6
        yaw = 90 * np.dot(pts[NARIZ] - (pts[BOCHECHA_E] + pts[BOCHECHA_D]) / 2, (pts[BOCHECHA_D] - pts[BOCHECHA_E]) / largura) / largura
        pitch = 90 * np.dot(pts[NARIZ] - (pts[TESTA] + pts[QUEIXO]) / 2, (pts[QUEIXO] - pts[TESTA]) / altura) / altura

    bs = {nome: 0.0 for nome in BLENDSHAPES}
    if resultado.face_blendshapes:
        categorias = resultado.face_blendshapes[0]
        for nome, i in BLENDSHAPES.items():
            achou = next((c.score for c in categorias if c.category_name == nome), None)
            bs[nome] = achou if achou is not None else (categorias[i].score if i < len(categorias) else 0.0)
    sorriso = (bs["mouthSmileLeft"] + bs["mouthSmileRight"]) / 2
    # "O": boca aberta E labios arredondados (uma gargalhada abre a boca mas nao arredonda)
    redondo = max(bs["mouthFunnel"], bs["mouthPucker"])
    boca_o = min(bs["jawOpen"] / 0.45, redondo / 0.30, 1.0)
    return np.array([yaw, pitch]), pts, {"sorriso": sorriso, "o": boca_o}


class Mapa:
    """Converte (yaw, pitch) em posicao na tela usando os 5 pontos calibrados (cada lado separado)."""

    def __init__(self, dados):
        self.d = dados
        self.d.setdefault("neutro", dict(NEUTRO_PADRAO))

    @staticmethod
    def _eixo(v, centro, menos, mais, alvo_menos, alvo_mais):
        t = (v - centro) / (mais - centro) if abs(mais - centro) > 1e-6 else 0
        if t >= 0:
            return 0.5 + (alvo_mais - 0.5) * t
        return 0.5 - (0.5 - alvo_menos) * (v - centro) / (menos - centro) if abs(menos - centro) > 1e-6 else 0.5

    def tela(self, ang, ganho, largura, altura):
        d = self.d
        nx = self._eixo(ang[0], d["centro"][0], d["esquerda"][0], d["direita"][0], 0.05, 0.95)
        ny = self._eixo(ang[1], d["centro"][1], d["cima"][1], d["baixo"][1], 0.06, 0.94)
        return np.array([(0.5 + (nx - 0.5) * ganho) * largura, (0.5 + (ny - 0.5) * ganho) * altura])

    def salvar(self):
        with open(CALIBRACAO, "w", encoding="utf-8") as f:
            json.dump(self.d, f)

    @staticmethod
    def carregar():
        if not os.path.exists(CALIBRACAO):
            return None
        with open(CALIBRACAO, encoding="utf-8") as f:
            return Mapa(json.load(f))


class Suave:
    """Suavizacao adaptativa: parado = sem tremer, movimento rapido = sem atraso."""

    def __init__(self):
        self.v = None

    def __call__(self, p):
        if self.v is None:
            self.v = p
            return p
        dist = np.linalg.norm(p - self.v)
        if dist < 3:
            return self.v
        alfa = min(max(dist / 120, 0.18), 0.7)
        self.v = self.v + alfa * (p - self.v)
        return self.v


class Expressao:
    """Liga/desliga com histerese e so muda depois de CONFIRMA frames (evita clique sem querer)."""

    def __init__(self, liga, desliga):
        self.liga, self.desliga = liga, desliga
        self.ativa = False
        self.contagem = 0

    def __call__(self, valor):
        quer = valor > (self.desliga if self.ativa else self.liga)
        self.contagem = self.contagem + 1 if quer != self.ativa else 0
        if self.contagem >= CONFIRMA:
            self.ativa, self.contagem = quer, 0
        return self.ativa


# ---------------------------------------------------------------- calibracao

def calibrar(captura, rosto, mp, mouse, espelhar, relogio, so_centro=False):
    """Mostra os pontos em tela cheia; pra cada um guarda a mediana do angulo da cabeca."""
    W, H = mouse.largura, mouse.altura
    cv.namedWindow(JANELA_CAL, cv.WINDOW_NORMAL)
    cv.setWindowProperty(JANELA_CAL, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    fundo = np.full((H, W, 3), 18, np.uint8)

    def ler():
        ok, frame = captura.read()
        if not ok:
            return False, None, None
        if espelhar:
            frame = cv.flip(frame, 1)
        ang, _, expr = lerRosto(detectar(rosto, mp, frame, relogio), frame.shape[1], frame.shape[0])
        return True, ang, expr

    def escrever(img, linhas):
        for i, linha in enumerate(linhas):
            tam = cv.getTextSize(linha, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv.putText(img, linha, ((W - tam[0]) // 2, H // 2 + 90 + i * 46), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                       (230, 230, 230), 2, cv.LINE_AA)

    while True:
        ok, ang, _ = ler()
        if not ok:
            return None
        tela = fundo.copy()
        escrever(tela, ["Aponte o NARIZ para cada ponto (vire a cabeca, sem exagero)",
                        "Rosto neutro: sem sorrir e de boca fechada",
                        "rosto encontrado" if ang is not None else "procurando seu rosto...",
                        "ESPACO comeca  |  ESC cancela"])
        cv.circle(tela, (W // 2, H // 2), 14, (255, 229, 0), 2, cv.LINE_AA)
        cv.imshow(JANELA_CAL, tela)
        tecla = cv.waitKey(1) & 0xFF
        if tecla == 27:
            cv.destroyWindow(JANELA_CAL)
            return None
        if tecla == 32 and ang is not None:
            break

    medidas, neutro = {}, []
    for nome, px, py in (PONTOS[:1] if so_centro else PONTOS):
        alvo = (int(px * W), int(py * H))
        leituras, inicio, duracao = [], time.time(), 2.0
        while time.time() - inicio < duracao:
            t = time.time() - inicio
            ok, ang, expr = ler()
            if not ok:
                return None
            if ang is not None and t > 0.9:     # espera a cabeca chegar
                leituras.append(ang)
                neutro.append([expr["sorriso"], expr["o"]])
            tela = fundo.copy()
            raio = int(34 - 24 * min(t / duracao, 1))
            cv.circle(tela, alvo, raio + 8, (255, 229, 0), 3, cv.LINE_AA)
            cv.circle(tela, alvo, 6, (255, 255, 255), -1, cv.LINE_AA)
            cv.imshow(JANELA_CAL, tela)
            if (cv.waitKey(1) & 0xFF) == 27:
                cv.destroyWindow(JANELA_CAL)
                return None
        if len(leituras) < 5:
            cv.destroyWindow(JANELA_CAL)
            print(f"perdi o rosto no ponto '{nome}'")
            return None
        medidas[nome] = np.median(np.array(leituras), axis=0).tolist()
    cv.destroyWindow(JANELA_CAL)
    if neutro:
        n = np.median(np.array(neutro), axis=0)
        medidas["neutro"] = {"sorriso": float(n[0]), "o": float(n[1])}
    return medidas


# ---------------------------------------------------------------- principal

def barra(img, x, y, rotulo, valor, limiar, ativa, cor):
    largura = 180
    cv.rectangle(img, (x, y), (x + largura, y + 14), (40, 40, 40), -1)
    cv.rectangle(img, (x, y), (x + int(largura * min(valor, 1)), y + 14), cor if ativa else (160, 160, 160), -1)
    lx = x + int(largura * min(limiar, 1))
    cv.line(img, (lx, y - 3), (lx, y + 17), (255, 255, 255), 2)
    cv.putText(img, rotulo, (x + largura + 10, y + 13), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Mouse pela cabeca, cliques pelo rosto.")
    parser.add_argument("--fonte", default="webcam", help="webcam ou numero da camera (precisa ver o rosto)")
    parser.add_argument("--calibrar", action="store_true", help="faz uma calibracao nova")
    parser.add_argument("--ganho", type=float, default=1.0, help="sensibilidade inicial (1.0 = a calibrada)")
    parser.add_argument("--sorriso", type=float, default=0.35, help="quanto sorrir alem do neutro pra clicar (0 a 1)")
    parser.add_argument("--boca", type=float, default=0.6, help="quanto fazer o 'O' pra clicar (0 a 1)")
    parser.add_argument("--sem-espelho", action="store_true")
    args = parser.parse_args()
    espelhar = not args.sem_espelho

    import mediapipe as mp
    mouse = Mouse()
    rosto = criarDetectorRosto()
    relogio = Relogio()
    try:
        captura, _ = abrirFonte(args.fonte)
    except OSError as e:
        raise SystemExit(f"nao conectou na fonte {args.fonte}: {e}")

    def calibracaoCompleta():
        medidas = calibrar(captura, rosto, mp, mouse, espelhar, relogio)
        if medidas is None:
            return None
        abertura_x = abs(medidas["direita"][0] - medidas["esquerda"][0])
        abertura_y = abs(medidas["baixo"][1] - medidas["cima"][1])
        if abertura_x < 4 or abertura_y < 3:
            print(f"a cabeca quase nao virou na calibracao ({abertura_x:.0f} x {abertura_y:.0f} graus); tente de novo")
            return None
        mapa = Mapa(medidas)
        mapa.salvar()
        print(f"calibrado: {abertura_x:.0f} graus de lado a lado, {abertura_y:.0f} graus de cima a baixo")
        return mapa

    mapa = None if args.calibrar else Mapa.carregar()
    if mapa is None:
        mapa = calibracaoCompleta()
        if mapa is None:
            raise SystemExit("sem calibracao nao da pra usar")

    def criarDetectores():
        neutro = mapa.d["neutro"]
        liga_s = min(neutro["sorriso"] + args.sorriso, 0.9)
        liga_o = min(max(neutro["o"] + 0.2, args.boca), 0.95)
        return (Expressao(liga_s, neutro["sorriso"] + args.sorriso * 0.55),
                Expressao(liga_o, liga_o * 0.6))

    sorriso, boca = criarDetectores()

    cv.namedWindow(JANELA, cv.WINDOW_NORMAL)
    cv.resizeWindow(JANELA, 480, 300)
    try:
        cv.setWindowProperty(JANELA, cv.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    suave = Suave()
    ganho = args.ganho
    ligado = True
    esquerdo_apertado = False
    boca_antes = False
    trava_ate = 0.0
    fps = 0.0
    print("sorria = esquerdo | boca em O = direito | +/- sensibilidade | R recalibra | O recentraliza | P liga/desliga | Q sai")

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

            ang, pts_rosto, expr = lerRosto(detectar(rosto, mp, frame, relogio), w, h)
            if ang is not None:
                # ---- rosto -> botoes
                fazendo_o = boca(expr["o"])
                sorrindo = sorriso(expr["sorriso"]) and not fazendo_o
                if ligado:
                    if sorrindo and not esquerdo_apertado:
                        mouse.apertar("esquerdo")
                        esquerdo_apertado = True
                        trava_ate = agora + TRAVA_CLIQUE
                    elif not sorrindo and esquerdo_apertado:
                        mouse.soltar("esquerdo")
                        esquerdo_apertado = False
                    if fazendo_o and not boca_antes:
                        mouse.clicar("direito")
                        trava_ate = agora + TRAVA_CLIQUE
                boca_antes = fazendo_o

                # ---- cabeca -> cursor (parado enquanto faz o "O": abrir a boca mexe a cabeca)
                alvo = suave(mapa.tela(ang, ganho, mouse.largura, mouse.altura))
                if ligado and agora >= trava_ate and not fazendo_o:
                    mouse.mover(*alvo)
            elif esquerdo_apertado:   # o rosto sumiu segurando o botao: solta pra nao travar
                mouse.soltar("esquerdo")
                esquerdo_apertado = False

            # ---- janelinha
            if pts_rosto is not None:
                nariz = pts_rosto[NARIZ]
                direcao = np.array([math.sin(math.radians(ang[0] - mapa.d["centro"][0])),
                                    math.sin(math.radians(ang[1] - mapa.d["centro"][1]))]) * 160
                cv.line(frame, tuple(nariz.astype(int)), tuple((nariz + direcao).astype(int)), (0, 237, 204), 3, cv.LINE_AA)
                cv.circle(frame, tuple(nariz.astype(int)), 6, (255, 255, 255), -1, cv.LINE_AA)
                barra(frame, 14, 62, "sorriso (esquerdo)", expr["sorriso"], sorriso.liga, sorriso.ativa, (0, 237, 204))
                barra(frame, 14, 88, "boca em O (direito)", expr["o"], boca.liga, boca.ativa, (109, 42, 255))
            gasto = max(time.time() - t0, 1e-3)
            fps = 0.9 * fps + 0.1 / gasto if fps else 1 / gasto
            if not ligado:
                rotulo, cor = "DESLIGADO (P liga)", (255, 255, 255)
            elif ang is None:
                rotulo, cor = "sem rosto", (150, 150, 150)
            elif esquerdo_apertado:
                rotulo, cor = "ESQUERDO (segurando)", (0, 237, 204)
            elif boca.ativa:
                rotulo, cor = "DIREITO", (109, 42, 255)
            else:
                rotulo, cor = "seguindo a cabeca", (255, 255, 255)
            for esp, c in ((5, (0, 0, 0)), (2, cor)):
                cv.putText(frame, rotulo, (14, 40), cv.FONT_HERSHEY_SIMPLEX, 1.1, c, esp, cv.LINE_AA)
            cv.putText(frame, f"{fps:.0f} FPS | sensibilidade {ganho:.1f} | +/-  R  O  P  Q", (14, h - 16),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
            cv.imshow(JANELA, frame)

            # ---- teclado
            tecla = cv.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27) or cv.getWindowProperty(JANELA, cv.WND_PROP_VISIBLE) < 1:
                break
            if tecla in (ord("+"), ord("=")):
                ganho = min(ganho + 0.1, 3.0)
            elif tecla in (ord("-"), ord("_")):
                ganho = max(ganho - 0.1, 0.3)
            elif tecla == ord("p"):
                ligado = not ligado
                if not ligado and esquerdo_apertado:
                    mouse.soltar("esquerdo")
                    esquerdo_apertado = False
            elif tecla == ord("r"):
                novo_mapa = calibracaoCompleta()
                if novo_mapa is not None:
                    mapa, suave = novo_mapa, Suave()
                    sorriso, boca = criarDetectores()
            elif tecla == ord("o"):
                medidas = calibrar(captura, rosto, mp, mouse, espelhar, relogio, so_centro=True)
                if medidas is not None:
                    # desloca os 5 pontos juntos pro novo centro
                    delta = np.array(medidas["centro"]) - np.array(mapa.d["centro"])
                    for nome, _, _ in PONTOS:
                        mapa.d[nome] = (np.array(mapa.d[nome]) + delta).tolist()
                    mapa.salvar()
                    suave = Suave()
                    print(f"recentralizado ({delta[0]:+.1f}, {delta[1]:+.1f}) graus")
    finally:
        if esquerdo_apertado:
            mouse.soltar("esquerdo")
        captura.release()
        rosto.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
