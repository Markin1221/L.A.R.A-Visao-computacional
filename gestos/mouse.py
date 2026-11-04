"""
Mouse controlado pela mao, pela webcam ou pela ESP32-CAM.

Gestos:
  mao aberta (qualquer outro gesto)        -> move o cursor (segue o centro da palma)
  pinca (polegar encostando no indicador)  -> botao ESQUERDO
        segurar a pinca = segurar o botao (da pra arrastar); abrir = soltar
  paz (indicador + medio em V)             -> clique com o botao DIREITO
  punho fechado                            -> pausa: o cursor fica parado

Teclas (com a janelinha da camera em foco):
  + / - sensibilidade (fica salva) | P liga/desliga o controle do mouse | Q ou ESC sai

Uso:
    pip install mediapipe
    python gestos/mouse.py
    python gestos/mouse.py --fonte 192.168.1.6 --sensibilidade 1.5
"""
import argparse
import ctypes
import json
import os
import sys
import time

import cv2 as cv
import numpy as np

from desenhar import ARTICULACAO, BASE_MEDIO, INDICADOR, MEDIO, ANELAR, MINDINHO, PALMA, POLEGAR, PULSO, \
    abrirFonte, baixarModelo, criarDetector

JANELA = "L.A.R.A | mouse por gestos"
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_mouse.json")
CONFIRMA = 3          # frames seguidos com o mesmo gesto antes de valer (evita clique fantasma)
TRAVA_CLIQUE = 0.25   # s com o cursor parado logo depois de apertar (formar o L mexe um pouco a mao)

CONEXOES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
            (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# ---------------------------------------------------------------- mouse do sistema

class Mouse:
    """No Windows usa a API do sistema direto (sem instalar nada); nos outros, o pynput."""

    def __init__(self):
        self.windows = sys.platform.startswith("win")
        if self.windows:
            self.user32 = ctypes.windll.user32
            try:   # coordenadas reais mesmo com zoom de tela (125%, 150%...)
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                self.user32.SetProcessDPIAware()
            self.largura = self.user32.GetSystemMetrics(0)
            self.altura = self.user32.GetSystemMetrics(1)
        else:
            from pynput.mouse import Button, Controller
            self.controle, self.botoes = Controller(), Button
            import tkinter
            raiz = tkinter.Tk()
            self.largura, self.altura = raiz.winfo_screenwidth(), raiz.winfo_screenheight()
            raiz.destroy()

    def mover(self, x, y):
        x, y = int(min(max(x, 0), self.largura - 1)), int(min(max(y, 0), self.altura - 1))
        if self.windows:
            self.user32.SetCursorPos(x, y)
        else:
            self.controle.position = (x, y)

    def _evento(self, codigo):
        self.user32.mouse_event(codigo, 0, 0, 0, 0)

    def apertar(self, botao):
        if self.windows:
            self._evento(0x0002 if botao == "esquerdo" else 0x0008)
        else:
            self.controle.press(self.botoes.left if botao == "esquerdo" else self.botoes.right)

    def soltar(self, botao):
        if self.windows:
            self._evento(0x0004 if botao == "esquerdo" else 0x0010)
        else:
            self.controle.release(self.botoes.left if botao == "esquerdo" else self.botoes.right)

    def clicar(self, botao):
        self.apertar(botao)
        self.soltar(botao)


# ---------------------------------------------------------------- gestos

PINCA_FECHA, PINCA_ABRE = 0.30, 0.45   # histerese: depois de apertar, precisa abrir mais pra soltar


def classificar(pts, anterior=None):
    """
    Devolve 'esquerdo' (pinca), 'direito' (paz), 'punho' ou 'mover'.
    Usa distancias, entao funciona com a mao girada. 'anterior' e o gesto do frame passado
    (deixa a pinca estavel enquanto voce arrasta).
    """
    palma = np.linalg.norm(pts[BASE_MEDIO] - pts[PULSO]) + 1e-6

    def esticado(dedo):
        return np.linalg.norm(pts[dedo] - pts[PULSO]) > 1.15 * np.linalg.norm(pts[ARTICULACAO[dedo]] - pts[PULSO])

    def dobrado(dedo):
        return np.linalg.norm(pts[dedo] - pts[PULSO]) < 1.0 * np.linalg.norm(pts[ARTICULACAO[dedo]] - pts[PULSO])

    # polegar pra fora: ponta longe da base do indicador e mais longe do pulso que a junta
    polegar_fora = (np.linalg.norm(pts[POLEGAR] - pts[5]) / palma > 0.55
                    and np.linalg.norm(pts[POLEGAR] - pts[PULSO]) > np.linalg.norm(pts[3] - pts[PULSO]))

    # punho vem primeiro: com a mao fechada o polegar encosta no indicador e pareceria pinca
    if all(dobrado(d) for d in (INDICADOR, MEDIO, ANELAR, MINDINHO)) and not polegar_fora:
        return "punho"
    pinca = np.linalg.norm(pts[POLEGAR] - pts[INDICADOR]) / palma
    if pinca < (PINCA_ABRE if anterior == "esquerdo" else PINCA_FECHA):
        return "esquerdo"
    if esticado(INDICADOR) and esticado(MEDIO) and not esticado(ANELAR) and not esticado(MINDINHO):
        return "direito"
    return "mover"


class Suavizador:
    """Media movel que se adapta: parado = bem suave (sem tremer); rapido = acompanha sem atraso."""

    def __init__(self):
        self.valor = None
        self.t = None

    def __call__(self, p, agora):
        p = np.asarray(p, float)
        if self.valor is None:
            self.valor, self.t = p, agora
            return p
        dt = max(agora - self.t, 1e-3)
        velocidade = np.linalg.norm(p - self.valor) / dt          # pixels de tela por segundo
        alfa = min(max(velocidade / 2500, 0.12), 0.75)
        self.valor = alfa * p + (1 - alfa) * self.valor
        self.t = agora
        return self.valor


# ---------------------------------------------------------------- principal

def main():
    parser = argparse.ArgumentParser(description="Mouse controlado por gestos da mao.")
    parser.add_argument("--fonte", default="webcam", help="webcam, numero da camera ou IP da ESP32")
    parser.add_argument("--sensibilidade", type=float,
                        help="maior = mexe menos a mao pra cruzar a tela (1.0 = a imagem inteira). Padrao: a ultima usada")
    parser.add_argument("--sem-espelho", action="store_true")
    args = parser.parse_args()

    mouse = Mouse()
    baixarModelo()
    mp, detector = criarDetector(num_maos=1)
    try:
        captura, _ = abrirFonte(args.fonte)
    except OSError as e:
        raise SystemExit(f"nao conectou na fonte {args.fonte}: {e}")

    cv.namedWindow(JANELA, cv.WINDOW_NORMAL)
    cv.resizeWindow(JANELA, 480, 300)
    try:
        cv.setWindowProperty(JANELA, cv.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    sensibilidade = args.sensibilidade
    if sensibilidade is None:
        try:
            with open(CONFIG, encoding="utf-8") as f:
                sensibilidade = float(json.load(f)["sensibilidade"])
        except Exception:
            sensibilidade = 1.4

    def areaAtiva(s):
        """Parte da camera que vira a tela inteira (centralizada); fora dela o cursor fica na borda."""
        lado = 1 / s
        return ((1 - lado) / 2, (1 - lado) / 2, (1 + lado) / 2, (1 + lado) / 2)

    area = areaAtiva(sensibilidade)

    suavizar = Suavizador()
    ligado = True
    gesto_visto, repeticoes, gesto = "mover", 0, "mover"
    esquerdo_apertado = False
    trava_ate = 0.0
    ultimo_ts = -1
    fps = 0.0
    print(f"tela {mouse.largura}x{mouse.altura} | sensibilidade {sensibilidade:.1f} | +/- ajusta | P liga/desliga | Q sai")

    try:
        while True:
            t0 = time.time()
            ok, frame = captura.read()
            if not ok:
                break
            if not args.sem_espelho:
                frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = np.ascontiguousarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            ts = max(int(time.monotonic() * 1000), ultimo_ts + 1)
            ultimo_ts = ts
            resultado = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            agora = time.time()

            if resultado.hand_landmarks:
                pts = np.array([[p.x * w, p.y * h] for p in resultado.hand_landmarks[0]])

                # gesto so muda depois de aparecer CONFIRMA frames seguidos
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

                # cursor segue o centro da palma (muda pouco quando voce faz os gestos)
                centro = pts[PALMA].mean(axis=0) / [w, h]
                nx = (centro[0] - area[0]) / (area[2] - area[0])
                ny = (centro[1] - area[1]) / (area[3] - area[1])
                alvo = suavizar([nx * mouse.largura, ny * mouse.altura], agora)
                if ligado and gesto != "punho" and agora >= trava_ate:
                    mouse.mover(*alvo)

                for a, b in CONEXOES:
                    cv.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (255, 229, 0), 2, cv.LINE_AA)
                cv.circle(frame, tuple((centro * [w, h]).astype(int)), 8, (255, 255, 255), -1, cv.LINE_AA)
            else:
                gesto_visto, repeticoes = "mover", 0
                if esquerdo_apertado:   # a mao sumiu segurando o botao: solta pra nao travar
                    mouse.soltar("esquerdo")
                    esquerdo_apertado = False
                gesto = "mover"

            # ---- janelinha de controle
            x0, y0, x1, y1 = (np.array(area) * [w, h, w, h]).astype(int)
            cv.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1, cv.LINE_AA)
            gasto = max(time.time() - t0, 1e-3)
            fps = 0.9 * fps + 0.1 / gasto if fps else 1 / gasto
            rotulo = {"esquerdo": "ESQUERDO (segurando)", "direito": "DIREITO", "punho": "pausado",
                      "mover": "movendo"}[gesto]
            if not ligado:
                rotulo = "DESLIGADO (P liga)"
            cor = {"esquerdo": (0, 237, 204), "direito": (109, 42, 255)}.get(gesto, (255, 255, 255))
            for espessura, c in ((5, (0, 0, 0)), (2, cor)):
                cv.putText(frame, rotulo, (14, 36), cv.FONT_HERSHEY_SIMPLEX, 1.0, c, espessura, cv.LINE_AA)
            cv.putText(frame, f"{fps:.0f} FPS | sensibilidade {sensibilidade:.1f} (+/-)", (14, h - 16),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
            cv.imshow(JANELA, frame)

            tecla = cv.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27) or cv.getWindowProperty(JANELA, cv.WND_PROP_VISIBLE) < 1:
                break
            if tecla in (ord("+"), ord("="), ord("-"), ord("_")):
                passo = 0.1 if tecla in (ord("+"), ord("=")) else -0.1
                sensibilidade = round(min(max(sensibilidade + passo, 1.0), 4.0), 1)
                area = areaAtiva(sensibilidade)
                try:
                    with open(CONFIG, "w", encoding="utf-8") as f:
                        json.dump({"sensibilidade": sensibilidade}, f)
                except OSError:
                    pass
            if tecla == ord("p"):
                ligado = not ligado
                if not ligado and esquerdo_apertado:
                    mouse.soltar("esquerdo")
                    esquerdo_apertado = False
    finally:
        if esquerdo_apertado:
            mouse.soltar("esquerdo")
        captura.release()
        detector.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
