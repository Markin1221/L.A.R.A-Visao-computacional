"""
L.A.R.A Estudio: desenho no ar + caixa de areia com fisica, tudo controlado pelas maos.
Troque de modo no botao do topo (aponte o indicador pra ele) ou com a tecla TAB.

MODO DESENHO
  pinca                                   -> desenha (traco que encosta em outro vira um desenho so)
  pinca com as DUAS maos num desenho      -> move e redimensiona
  rock (indicador + mindinho) e girar     -> gira o desenho
  punho                                   -> borracha
  indicador numa cor / pinca na barra     -> troca a cor (tem arco-iris) / espessura
  botao Espelho                           -> desenha espelhado (simetria)

MODO FISICA
  punho (ou pinca, botao Segurar)         -> pega objeto da aba, objeto ou DESENHO; soltar mexendo = joga
  duas maos no objeto                     -> estica; solta uma e depois a outra = estilingue
  mao aberta passando rapido              -> tapa  |  rock -> ima
  desenhos viram rampas/paredes; pegar um desenho solta ele (cai e da pra jogar)
  bomba explode 2.5 s depois de soltar

ROSTO (botao Rosto, so quando voce quiser)
  desenho solto em cima do rosto gruda nele e segue a cabeca (oculos, chapeu, bigode...)
  pra desgrudar: pegue com as duas maos e solte fora do rosto
  na fisica: a cabeca rebate os objetos e a boca aberta come eles

Teclas: TAB modo | O rosto | Z desfaz | C limpa | K salva foto | F tela cheia | H ajuda | Q sai
        desenho: E espelho | fisica: P punho/pinca, G gravidade, L camera lenta, R rastro, X fixa desenhos, A aba

Uso:
    pip install mediapipe pymunk pillow
    python gestos/estudio.py
    python gestos/estudio.py --fonte http://192.168.0.103:4747/video
"""
import argparse
import colorsys
import math
import os
import time
import urllib.request
from collections import deque

import cv2 as cv
import numpy as np
import pymunk

from desenhar import (CORES, DESCARTA_TRACO, ESP_MAX, ESP_MIN, ESPERA_COR, ESPERA_DESENHO, PASTA, SUMIU,
                      abrirFonte, angulo_limpo, aplicar, associarMaos, baixarModelo, criarDetector, escala,
                      rotacao, translacao)
from fisica import (ALCANCE, ALCANCE_2A_MAO, BOMBA_FORCA, BOMBA_PAVIO, BOMBA_RAIO, CAT_MAO, CAT_OBJETO,
                    GRAVIDADE, IMA_FORCA, IMA_RAIO, MAX_OBJETOS, MODOS_GRAVIDADE, PERDIDA_MAX, SOLTAR_ESPERA,
                    SUBPASSOS, TAPA_VEL, TIPOS, VEL_MAX, MaoFisica, Objeto, criarParedes)

JANELA = "L.A.R.A | estudio"
CAT_DESENHO, CAT_ROSTO = 0b1000, 0b10000
PEGAVEIS = pymunk.ShapeFilter(mask=CAT_OBJETO | CAT_DESENHO)
ESPERA_BOTAO = 0.45
ARCO = None   # "cor" do pincel arco-iris

MODELO_ROSTO = os.path.join(PASTA, "face_landmarker.task")
URL_ROSTO = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# visual
VIDRO = (24, 20, 18)        # fundo dos paineis (BGR)
TEXTO = (245, 245, 245)
APAGADO = (170, 170, 170)
COR_FISICA = (60, 190, 255)    # ambar
COR_ROSTO = (230, 140, 255)    # rosa


# ================================================================ texto bonito (Pillow) com cache

class Escritor:
    def __init__(self):
        self.cache = {}
        self.fontes = {}
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.Image, self.ImageDraw, self.ImageFont = Image, ImageDraw, ImageFont
            candidatos = {False: ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
                          True: ["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeuib.ttf",
                                 "C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]}
            self.caminho = {k: next((c for c in v if os.path.exists(c)), None) for k, v in candidatos.items()}
            self.ok = any(self.caminho.values())
        except ImportError:
            self.ok = False

    def _fonte(self, tam, negrito):
        chave = (tam, negrito)
        if chave not in self.fontes:
            caminho = self.caminho[negrito] or self.caminho[not negrito]
            self.fontes[chave] = self.ImageFont.truetype(caminho, tam)
        return self.fontes[chave]

    def _mascara(self, msg, tam, negrito):
        chave = (msg, tam, negrito)
        if chave not in self.cache:
            if len(self.cache) > 600:
                self.cache.clear()
            f = self._fonte(tam, negrito)
            x0, y0, x1, y1 = f.getbbox(msg or " ")
            im = self.Image.new("L", (max(x1 - x0, 1) + 4, max(y1 - y0, 1) + 4), 0)
            self.ImageDraw.Draw(im).text((2 - x0, 2 - y0), msg, font=f, fill=255)
            a = np.asarray(im, np.float32) / 255
            sombra = cv.GaussianBlur(a, (0, 0), 1.6)
            self.cache[chave] = (a, sombra)
        return self.cache[chave]

    def largura(self, msg, tam=18, negrito=False):
        if not self.ok:
            return cv.getTextSize(msg, cv.FONT_HERSHEY_SIMPLEX, tam / 30, 1)[0][0]
        return self._mascara(msg, tam, negrito)[0].shape[1]

    def escrever(self, img, msg, pos, tam=18, cor=TEXTO, negrito=False, centro=False, sombra=True):
        """pos = canto de cima-esquerda (ou o centro, com centro=True)."""
        if not self.ok:
            esc = tam / 30
            (lw, lh), _ = cv.getTextSize(msg, cv.FONT_HERSHEY_SIMPLEX, esc, 1)
            x, y = (pos[0] - lw // 2, pos[1] + lh // 2) if centro else (pos[0], pos[1] + lh)
            cv.putText(img, msg, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, esc, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(img, msg, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, esc, cor, 1 + negrito, cv.LINE_AA)
            return lw
        a, s = self._mascara(msg, tam, negrito)
        hh, ww = a.shape
        x, y = int(pos[0]), int(pos[1])
        if centro:
            x, y = x - ww // 2, y - hh // 2
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + ww, W), min(y + hh, H)
        if x1 <= x0 or y1 <= y0:
            return ww
        a = a[y0 - y:y1 - y, x0 - x:x1 - x, None]
        roi = img[y0:y1, x0:x1].astype(np.float32)
        if sombra:
            roi *= 1 - 0.7 * s[y0 - y:y1 - y, x0 - x:x1 - x, None]
        roi = roi * (1 - a) + np.array(cor, np.float32) * a
        img[y0:y1, x0:x1] = roi.astype(np.uint8)
        return ww


def painel(img, x0, y0, x1, y1, raio=14, cor=VIDRO, alfa=0.62, borda=None):
    """Retangulo arredondado translucido (so mexe na regiao dele, rapido)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(max(x0, 0)), int(max(y0, 0)), int(min(x1, W)), int(min(y1, H))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = img[y0:y1, x0:x1]
    mascara = np.zeros(roi.shape[:2], np.uint8)
    r = int(min(raio, (x1 - x0) // 2, (y1 - y0) // 2))
    w, h = x1 - x0, y1 - y0
    cv.rectangle(mascara, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv.rectangle(mascara, (0, r), (w - 1, h - r - 1), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv.circle(mascara, (cx, cy), r, 255, -1, cv.LINE_AA)
    a = (mascara.astype(np.float32) / 255 * alfa)[..., None]
    roi[:] = (roi * (1 - a) + np.array(cor, np.float32) * a).astype(np.uint8)
    if borda is not None:
        cv.rectangle(img, (x0 + r, y0), (x1 - r, y0), borda, 1, cv.LINE_AA)


# ================================================================ sprites dos objetos

def _esfera(r):
    lado = 2 * r + 4
    c = lado / 2
    yy, xx = np.mgrid[0:lado, 0:lado].astype(np.float32) + 0.5
    nx, ny = (xx - c) / r, (yy - c) / r
    d2 = nx * nx + ny * ny
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    L = np.array([-0.45, -0.6, 0.66])
    L /= np.linalg.norm(L)
    dif = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    Hv = L + [0, 0, 1]
    Hv /= np.linalg.norm(Hv)
    spec = np.clip(nx * Hv[0] + ny * Hv[1] + nz * Hv[2], 0, 1) ** 70
    alfa = np.clip((1 - np.sqrt(d2)) * r + 0.5, 0, 1)
    borda = np.clip((np.sqrt(d2) - 0.82) / 0.18, 0, 1) * 0.25   # escurece a borda (volume)
    sombra = np.clip(0.62 - 0.75 * dif + borda, 0, 0.8) * alfa
    brilho = np.clip(spec * 0.95, 0, 1) * alfa
    return lado, alfa, np.dstack([sombra, brilho]).astype(np.float32)


def criarSprite(tipo):
    """(rgba, luz[sombra, brilho], luz_gira) desenhados uma vez so; depois so sao transformados."""
    t = TIPOS[tipo]
    r = t["tamanho"]
    if t["forma"] == "caixa":
        lado = 2 * r + 4
        img = np.zeros((lado, lado, 3), np.uint8)
        img[:] = (70, 135, 200)
        b = max(3, int(r * 0.2))
        escuro, medio = (35, 80, 130), (52, 108, 165)
        for k in range(1, 4):   # tabuas
            y = int(2 + k * (lado - 4) / 4)
            cv.line(img, (2, y), (lado - 3, y), medio, 2, cv.LINE_AA)
        cv.line(img, (2 + b, 2 + b), (lado - 3 - b, lado - 3 - b), escuro, b, cv.LINE_AA)   # travessa
        cv.rectangle(img, (2, 2), (lado - 3, lado - 3), escuro, b)
        for cx, cy in ((b + 3, b + 3), (lado - b - 4, b + 3), (b + 3, lado - b - 4), (lado - b - 4, lado - b - 4)):
            cv.circle(img, (cx, cy), max(2, b // 3), (150, 170, 190), -1, cv.LINE_AA)
        alfa = np.zeros((lado, lado), np.float32)
        cv.rectangle(alfa, (2, 2), (lado - 3, lado - 3), 1.0, -1)
        yy, xx = np.mgrid[0:lado, 0:lado].astype(np.float32) / lado
        sombra = np.clip((xx + yy) * 0.22 - 0.05, 0, 0.4) * alfa
        brilho = np.clip(0.18 - yy * 0.6, 0, 0.18) * alfa
        luz = np.dstack([sombra, brilho]).astype(np.float32)
        return np.dstack([img, (alfa * 255).astype(np.uint8)]), luz, True

    lado, alfa, luz = _esfera(r)
    c = lado // 2
    img = np.zeros((lado, lado, 3), np.uint8)
    esp = max(2, r // 12)
    if tipo == "bola":   # basquete
        img[:] = (40, 118, 238)
        cor = (35, 45, 80)
        cv.line(img, (0, c), (lado, c), cor, esp, cv.LINE_AA)
        cv.line(img, (c, 0), (c, lado), cor, esp, cv.LINE_AA)
        cv.ellipse(img, (int(c - 1.05 * r), c), (int(0.7 * r), int(1.1 * r)), 0, -70, 70, cor, esp, cv.LINE_AA)
        cv.ellipse(img, (int(c + 1.05 * r), c), (int(0.7 * r), int(1.1 * r)), 0, 110, 250, cor, esp, cv.LINE_AA)
    elif tipo == "superbola":
        img[:] = (80, 215, 95)
        cv.circle(img, (c, c), int(r * 0.62), (60, 235, 250), max(3, r // 5), cv.LINE_AA)
        cv.circle(img, (c, c), int(r * 0.2), (255, 120, 60), -1, cv.LINE_AA)
    elif tipo == "boliche":
        img[:] = (120, 45, 70)
        rng = np.random.default_rng(3)
        for _ in range(7):   # marmorizado
            p = rng.uniform(0, lado, 2).astype(int)
            cv.ellipse(img, tuple(p), (int(r * rng.uniform(0.3, 0.8)), int(r * 0.12)), rng.uniform(0, 180),
                       0, 360, (160, 80, 110), 2, cv.LINE_AA)
        for k in range(3):
            a = -1.6 + (k - 1) * 0.42
            p = (c + r * 0.5 * math.cos(a), c + r * 0.5 * math.sin(a) + (k == 1) * -3)
            cv.circle(img, (int(p[0]), int(p[1])), max(3, int(r * 0.12)), (25, 12, 20), -1, cv.LINE_AA)
    elif tipo == "balao":
        img[:] = (70, 55, 235)
        cv.ellipse(img, (c, c), (int(r * 0.55), int(r * 0.85)), 0, 200, 250, (140, 130, 255), max(2, r // 10), cv.LINE_AA)
    elif tipo == "bomba":
        img[:] = (55, 50, 58)
        tampa = np.zeros_like(alfa)
        cv.rectangle(tampa, (int(c - 0.32 * r), int(c - 1.0 * r)), (int(c + 0.32 * r), int(c - 0.72 * r)), 1.0, -1)
        img[tampa > 0] = (120, 120, 130)
        alfa = np.maximum(alfa, tampa)
    else:
        img[:] = t["cor"]
    return np.dstack([img, (np.clip(alfa, 0, 1) * 255).astype(np.uint8)]), luz, False


def colar(img, rgba, A, luz=None, A_luz=None, sombra=True):
    """Cola o sprite com a transformacao afim A (2x3, coords do sprite -> tela), com sombra e luz."""
    h, w = rgba.shape[:2]
    cantos = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) @ A[:, :2].T + A[:, 2]
    H, W = img.shape[:2]
    x0, y0 = np.floor(cantos.min(0)).astype(int) - 14
    x1, y1 = np.ceil(cantos.max(0)).astype(int) + 14
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    desloca = np.array([x0, y0], float)
    B = A.copy()
    B[:, 2] -= desloca
    spr = cv.warpAffine(rgba, B, (x1 - x0, y1 - y0), flags=cv.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = spr[..., 3].astype(np.float32) / 255
    roi = img[y0:y1, x0:x1].astype(np.float32)
    if sombra:
        s = np.zeros_like(a)
        s[10:, 7:] = a[:-10, :-7]
        s = cv.GaussianBlur(s, (0, 0), 5) * 0.5 * (1 - a)
        roi *= (1 - s)[..., None]
    a3 = a[..., None]
    roi = roi * (1 - a3) + spr[..., :3].astype(np.float32) * a3
    if luz is not None:
        Bl = (A_luz if A_luz is not None else A).copy()
        Bl[:, 2] -= desloca
        L = cv.warpAffine(luz, Bl, (x1 - x0, y1 - y0), flags=cv.INTER_LINEAR)
        roi *= 1 - L[..., 0:1]
        roi = roi * (1 - L[..., 1:2]) + 255 * L[..., 1:2]
    img[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def afim(L, pos, centro):
    A = np.zeros((2, 3))
    A[:, :2] = L
    A[:, 2] = np.asarray(pos, float) - L @ np.asarray(centro, float)
    return A


def esticamento(s, a):
    s = min(max(s, 0.4), 3.0)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return R @ np.diag([s, 1 / math.sqrt(s)]) @ R.T


# ================================================================ desenhos (grupos de tracos)

class Traco:
    def __init__(self, pts, cor, esp):
        self.pts = [list(p) for p in pts]   # coordenadas proprias do desenho
        self.cor = cor                      # BGR ou ARCO
        self.esp = esp


class Desenho:
    """Varios tracos que se encostam = um desenho so. M leva pra tela. local != None: grudado no rosto."""

    def __init__(self, tracos=None, M=None, local=None):
        self.tracos = tracos or []
        self.M = np.eye(3) if M is None else M.copy()
        self.local = None if local is None else local.copy()
        self.corpo = None

    def fator(self):
        return math.sqrt(abs(np.linalg.det(self.M[:2, :2])))

    def telaDe(self, t):
        return aplicar(self.M, t.pts)

    def tela(self):
        return np.vstack([self.telaDe(t) for t in self.tracos]) if self.tracos else np.zeros((0, 2))

    def centro(self):
        return self.tela().mean(axis=0)


def densificar(pts, passo=5.0):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts
    saida = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(np.linalg.norm(b - a) // passo)
        saida.extend(a + (b - a) * k / (n + 1) for k in range(1, n + 1))
        saida.append(b)
    return np.array(saida)


def encostam(a, ea, b, eb):
    """Dois tracos (pontos na mesma escala) se tocam ou se cruzam?"""
    folga = (ea + eb) / 2 + 3
    if (a.min(0) - folga > b.max(0)).any() or (b.min(0) - folga > a.max(0)).any():
        return False
    A, B = densificar(a, 4), densificar(b, 4)
    for i in range(0, len(A), 400):
        d2 = ((A[i:i + 400, None, :] - B[None, :, :]) ** 2).sum(-1)
        if d2.min() <= folga * folga:
            return True
    return False


def componentes(tracos):
    """Separa tracos em grupos conectados (tudo nas coords do mesmo desenho)."""
    pai = list(range(len(tracos)))

    def raiz(i):
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = pai[i]
        return i

    arr = [np.array(t.pts, float) for t in tracos]
    for i in range(len(tracos)):
        for j in range(i + 1, len(tracos)):
            if raiz(i) != raiz(j) and encostam(arr[i], tracos[i].esp, arr[j], tracos[j].esp):
                pai[raiz(i)] = raiz(j)
    grupos = {}
    for i, t in enumerate(tracos):
        grupos.setdefault(raiz(i), []).append(t)
    return list(grupos.values())


def dist_pts(p, pts):
    if len(pts) == 1:
        return float(np.linalg.norm(pts[0] - p))
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    return float(np.min(np.linalg.norm(a + ab * t[:, None] - p, axis=1)))


def selecionar(desenhos, p):
    dentro, perto, melhor = [], None, 1e9
    for d in desenhos:
        t = d.tela()
        if not len(t):
            continue
        folga = 25
        lo, hi = t.min(0) - folga, t.max(0) + folga
        if (p >= lo).all() and (p <= hi).all():
            dentro.append((float(np.prod(hi - lo)), d))
        dd = min(dist_pts(p, d.telaDe(tr)) for tr in d.tracos)
        if dd < melhor:
            melhor, perto = dd, d
    if dentro:
        return min(dentro, key=lambda x: x[0])[1]
    return perto if melhor < 80 else None


def apagar(desenhos, centro, raio):
    novos = []
    for d in desenhos:
        f = d.fator()
        inversa = np.linalg.inv(d.M)
        mudou, sobrou = False, []
        for tr in d.tracos:
            t = d.telaDe(tr)
            r = raio + tr.esp * f / 2
            if dist_pts(centro, t) > r:
                sobrou.append(tr)
                continue
            denso = densificar(t, 4)
            fora = np.linalg.norm(denso - centro, axis=1) > r
            mudou = True
            i = 0
            while i < len(denso):
                if not fora[i]:
                    i += 1
                    continue
                j = i
                while j < len(denso) and fora[j]:
                    j += 1
                if j - i >= 2:
                    sobrou.append(Traco(aplicar(inversa, denso[i:j]), tr.cor, tr.esp))
                i = j
        if not mudou:
            novos.append(d)
            continue
        for grupo in componentes(sobrou):   # o que a borracha separou vira desenho separado
            novos.append(Desenho(grupo, d.M, d.local))
    return novos


def corArco(i, agora):
    r, g, b = colorsys.hsv_to_rgb(((i * 0.012) + agora * 0.25) % 1, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def desenharDesenhos(frame, desenhos, agora, destaque=None):
    """Linhas neon: brilho borrado em meia resolucao (rapido) + linha + nucleo branco."""
    linhas = []
    for d in desenhos:
        f = d.fator()
        for tr in d.tracos:
            if len(tr.pts) < 2:
                continue
            linhas.append((d, tr, d.telaDe(tr).round().astype(np.int32), max(1, int(round(tr.esp * f)))))
    if not linhas:
        return frame
    h, w = frame.shape[:2]
    brilho = np.zeros((h // 2, w // 2, 3), np.uint8)
    for d, tr, pts, esp in linhas:
        meio = (pts // 2).astype(np.int32)
        if tr.cor is ARCO:
            for i in range(0, len(meio) - 1, 3):
                cv.line(brilho, tuple(meio[i]), tuple(meio[min(i + 3, len(meio) - 1)]), corArco(i, agora),
                        max(2, esp * 3 // 2), cv.LINE_AA)
        else:
            cv.polylines(brilho, [meio], False, tr.cor, max(2, esp * 3 // 2), cv.LINE_AA)
    brilho = cv.GaussianBlur(brilho, (0, 0), 5)
    saida = cv.add(frame, cv.resize(brilho, (w, h), interpolation=cv.INTER_LINEAR))
    for d, tr, pts, esp in linhas:
        if d is destaque:
            cv.polylines(saida, [pts], False, (255, 255, 255), esp + 8, cv.LINE_AA)
        if tr.cor is ARCO:
            for i in range(len(pts) - 1):
                cv.line(saida, tuple(pts[i]), tuple(pts[i + 1]), corArco(i, agora), esp, cv.LINE_AA)
        else:
            cv.polylines(saida, [pts], False, tr.cor, esp, cv.LINE_AA)
        cv.polylines(saida, [pts], False, (255, 255, 255), max(1, esp // 3), cv.LINE_AA)
    return saida


# ================================================================ rosto

def criarDetectorRosto():
    if not os.path.exists(MODELO_ROSTO):
        print("baixando o modelo de rosto do MediaPipe (uns 4 MB)...")
        urllib.request.urlretrieve(URL_ROSTO, MODELO_ROSTO)
    from mediapipe.tasks.python import BaseOptions, vision
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELO_ROSTO), running_mode=vision.RunningMode.VIDEO,
        num_faces=1, output_face_blendshapes=True))


class Rosto:
    """Onde esta o rosto: um 'sistema de coordenadas' (F) que segue os olhos, o contorno e a boca."""

    def __init__(self):
        self.detector = None
        self.ts = -1
        self.visivel = False
        self.visto = 0.0
        self.F = np.eye(3)
        self.suave = None
        self.aberta = False

    def atualizar(self, mp, rgb, w, h, agora):
        if self.detector is None:
            self.detector = criarDetectorRosto()
        self.ts = max(int(time.monotonic() * 1000), self.ts + 1)
        res = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.ts)
        if not res.face_landmarks:
            if agora - self.visto > 0.4:
                self.visivel, self.suave = False, None
            return
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
        if len(pts) >= 478:
            e1, e2 = pts[468], pts[473]
        else:
            e1, e2 = pts[[33, 133]].mean(0), pts[[362, 263]].mean(0)
        if e1[0] > e2[0]:
            e1, e2 = e2, e1
        origem, v = (e1 + e2) / 2, e2 - e1
        ang, d = math.atan2(v[1], v[0]), max(np.linalg.norm(v), 1.0)
        if self.suave is None:
            self.suave = [origem, ang, d]
        else:
            o0, a0, d0 = self.suave
            a = 0.55
            self.suave = [a * origem + (1 - a) * o0, a0 + a * angulo_limpo(ang - a0), a * d + (1 - a) * d0]
        o, ang, d = self.suave
        self.F = translacao(o) @ rotacao(ang) @ escala(d / 100)
        self.olhos, self.dist_olhos, self.angulo = o, d, ang
        self.oval = pts[OVAL]
        self.centro_oval = self.oval.mean(0)
        self.boca = pts[[13, 14]].mean(0)
        self.boca_larg = np.linalg.norm(pts[61] - pts[291])
        jaw = 0.0
        if res.face_blendshapes:
            jaw = next((c.score for c in res.face_blendshapes[0] if c.category_name == "jawOpen"), 0.0)
        abertura = np.linalg.norm(pts[13] - pts[14]) / (self.boca_larg + 1e-6)
        self.aberta = (jaw > (0.2 if self.aberta else 0.32)) or abertura > (0.35 if self.aberta else 0.5)
        self.visivel, self.visto = True, agora

    def contem(self, p, folga=1.5):
        if not self.visivel:
            return False
        oval = (self.oval - self.centro_oval) * folga + self.centro_oval
        return cv.pointPolygonTest(oval.astype(np.float32), (float(p[0]), float(p[1])), False) >= 0

    def fechar(self):
        if self.detector is not None:
            self.detector.close()


# ================================================================ fisica: objetos e desenhos com corpo

class Coisa(Objeto):
    def __init__(self, espaco, tipo, pos):
        super().__init__(espaco, tipo, pos)
        self.massa = self.t["massa"]
        self.raio = self.t["tamanho"]

    @property
    def shapes(self):
        return [self.shape]


class CorpoDesenho:
    """Um desenho no mundo da fisica: parado (rampa/parede) ate alguem pegar; ai cai e pode ser jogado."""

    def __init__(self, espaco, d):
        self.d = d
        tela = d.tela()
        self.c0 = tela.mean(axis=0)
        self.M0 = d.M.copy()
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = tuple(self.c0)
        self.shapes = []
        f = d.fator()
        massa = 0.0
        for tr in d.tracos:
            pts = d.telaDe(tr)
            if len(pts) >= 3:
                pts = cv.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), 2.0, False).reshape(-1, 2)
            r = max(tr.esp * f / 2, 3.0)
            pares = list(zip(pts[:-1], pts[1:])) or [(pts[0], pts[0])]
            for a, b in pares:
                s = pymunk.Segment(self.body, tuple(a - self.c0), tuple(b - self.c0), r)
                s.mass = max(np.linalg.norm(b - a) * r * 0.0015, 0.01)
                massa += s.mass
                s.elasticity, s.friction = 0.45, 0.9
                s.filter = pymunk.ShapeFilter(categories=CAT_DESENHO)
                self.shapes.append(s)
        espaco.add(self.body, *self.shapes)
        self.massa = max(massa, 0.3)
        self.raio = float(np.max(np.linalg.norm(tela - self.c0, axis=1))) if len(tela) else 20.0
        cor = next((tr.cor for tr in d.tracos if tr.cor is not ARCO), (255, 255, 255))
        self.t = {"massa": self.massa, "tamanho": self.raio, "quique": 0.45, "cor": cor}
        self.estado, self.maos, self.restricao = "livre", [], None
        self.esticar, self.balanco, self.pavio = (1.0, 0.0), None, None
        self.rastro = deque(maxlen=14)
        self.tipo = "desenho"

    @property
    def pos(self):
        return np.array(self.body.position)

    def atualizarPose(self):
        a = self.body.angle
        self.d.M = translacao(self.pos) @ rotacao(a) @ translacao(-self.c0) @ self.M0


# ================================================================ botoes

class Botao:
    def __init__(self, nome, rect, rotulo, acao, cor=None):
        self.nome, self.rect, self.rotulo, self.acao = nome, rect, rotulo, acao
        self.cor = cor

    def contem(self, p):
        x0, y0, x1, y1 = self.rect
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


# ================================================================ app

class Estudio:
    def __init__(self, args):
        self.args = args
        self.w = self.h = None
        self.modo = "desenho"
        self.escritor = Escritor()
        self.ajuda = True
        self.aviso = None   # (texto, t0)
        # rosto
        self.rosto_on = False
        self.rosto = Rosto()
        # desenho
        self.desenhos = []
        self.atual = []           # tracos sendo desenhados agora (2 com espelho): listas de pontos na tela
        self.inicio_atual = 0.0
        self.gesto_d, self.gesto = "livre", {}
        self.cor_idx, self.esp = 0, 8
        self.espelho = False
        self.mira_cor = {"idx": None, "desde": 0.0}
        # fisica
        self.espaco = pymunk.Space()
        self.espaco.damping = 0.995
        self.paredes = None
        self.coisas, self.corpos = [], []
        self.fisicas = {}
        self.dono = {}            # shape -> objeto (Coisa ou CorpoDesenho)
        self.pegar_com = "punho"
        self.modo_g, self.lenta, self.rastro, self.aba_aberta = 0, False, True, True
        self.particulas, self.explosoes = [], []
        self.cabeca = None
        self.comidos = 0
        # botoes
        self.mira_botao = {"nome": None, "desde": 0.0}
        self.sprites = {tipo: criarSprite(tipo) for tipo in TIPOS}

    # ------------------------------------------------------------ layout
    def montar(self, w, h):
        self.w, self.h = w, h
        self.paredes = criarParedes(self.espaco, w, h)
        self.cores = CORES + [ARCO]
        passo = min(64, (h - 300) / len(self.cores))
        self.pos_cores = [(56, int(146 + passo * i)) for i in range(len(self.cores))]
        self.barra = (128, 140, int(self.pos_cores[-1][1] + 10))
        passo = min(110, (h - 150) // len(TIPOS))
        self.raio_item = min(46, passo // 2)
        self.itens = [(tipo, (82, int(112 + passo / 2 + i * passo))) for i, tipo in enumerate(TIPOS)]
        cx = w // 2
        self.botoes_topo = [
            Botao("desenho", (cx - 250, 14, cx - 90, 58), lambda: "Desenho", lambda: self.trocarModo("desenho")),
            Botao("fisica", (cx - 80, 14, cx + 80, 58), lambda: "Fisica", lambda: self.trocarModo("fisica")),
            Botao("rosto", (cx + 90, 14, cx + 250, 58),
                  lambda: "Rosto: " + ("ligado" if self.rosto_on else "desligado"), self.trocarRosto, COR_ROSTO),
        ]
        x0, x1 = w - 250, w - 20
        self.botoes_desenho = [
            Botao("espelho", (x0, 80, x1, 124), lambda: "Espelho: " + ("ligado" if self.espelho else "desligado"),
                  self.trocarEspelho),
        ]
        self.botoes_fisica = [
            Botao("segurar", (x0, 80, x1, 124), lambda: "Segurar: " + self.pegar_com, self.trocarPegar),
            Botao("gravidade", (x0, 134, x1, 178), lambda: "Gravidade: " + MODOS_GRAVIDADE[self.modo_g][0],
                  self.trocarGravidade),
            Botao("fixar", (x0, 188, x1, 232), lambda: "Fixar desenhos", self.fixarDesenhos),
        ]

    def botoes(self):
        return self.botoes_topo + (self.botoes_desenho if self.modo == "desenho" else self.botoes_fisica)

    def avisar(self, msg):
        self.aviso = (msg, time.time())

    # ------------------------------------------------------------ acoes
    def trocarModo(self, modo=None):
        modo = modo or ("fisica" if self.modo == "desenho" else "desenho")
        if modo == self.modo:
            return
        self.terminarTraco()
        self.gesto_d, self.gesto = "livre", {}
        if modo == "fisica":
            self.entrarFisica()
        else:
            self.sairFisica()
        self.modo = modo
        self.avisar("modo " + ("fisica" if modo == "fisica" else "desenho"))

    def trocarRosto(self):
        self.rosto_on = not self.rosto_on
        if not self.rosto_on:
            self.tirarCabeca()
        self.avisar("rosto " + ("ligado: solte um desenho no rosto pra grudar" if self.rosto_on else "desligado"))

    def trocarEspelho(self):
        self.espelho = not self.espelho

    def trocarPegar(self):
        self.pegar_com = "pinca" if self.pegar_com == "punho" else "punho"
        self.soltarTudo()

    def trocarGravidade(self):
        self.modo_g = (self.modo_g + 1) % len(MODOS_GRAVIDADE)

    def fixarDesenhos(self):
        for c in self.corpos:
            if c.estado != "livre":
                self.soltarLivre(c)
            c.body.velocity, c.body.angular_velocity = (0, 0), 0
            c.body.body_type = pymunk.Body.STATIC
        self.avisar("desenhos fixos")

    def desfazer(self):
        if self.modo == "desenho" and self.desenhos:
            self.desenhos.pop()
        elif self.modo == "fisica" and self.coisas:
            self.removerCoisa(self.coisas[-1])

    def limpar(self):
        if self.modo == "desenho":
            self.desenhos.clear()
            self.atual = []
        else:
            self.soltarTudo()
            for c in list(self.coisas):
                self.removerCoisa(c)

    # ------------------------------------------------------------ desenho: tracos e grupos
    def terminarTraco(self, descartar=False):
        if not descartar:
            for pts, cor, esp in self.atual:
                if len(pts) >= 2:
                    self.incorporar(np.array(pts, float), cor, esp)
        self.atual = []

    def incorporar(self, tela, cor, esp):
        """Traco novo: se encosta em algum desenho, junta tudo num desenho so."""
        tocados = []
        for d in self.desenhos:
            f = d.fator()
            if any(encostam(tela, esp, d.telaDe(tr), tr.esp * f) for tr in d.tracos):
                tocados.append(d)
        if not tocados:
            novo = Desenho([Traco(tela, cor, esp)])
            self.desenhos.append(novo)
            self.grudarSePrecisar(novo)
            return
        base = max(tocados, key=lambda d: sum(len(t.pts) for t in d.tracos))
        inv = np.linalg.inv(base.M)
        fb = base.fator()
        for d in tocados:
            if d is base:
                continue
            conv = inv @ d.M
            r = d.fator() / fb
            base.tracos += [Traco(aplicar(conv, t.pts), t.cor, t.esp * r) for t in d.tracos]
            self.desenhos.remove(d)
        base.tracos.append(Traco(aplicar(inv, tela), cor, esp / fb))
        self.grudarSePrecisar(base)

    def grudarSePrecisar(self, d):
        """Com o rosto ligado: desenho que termina em cima do rosto gruda; fora dele, desgruda."""
        if not (self.rosto_on and self.rosto.visivel) or not d.tracos:
            return
        antes = d.local is not None
        d.local = np.linalg.inv(self.rosto.F) @ d.M if self.rosto.contem(d.centro()) else None
        if d.local is not None and not antes:
            self.avisar("grudou no rosto!")
        elif d.local is None and antes:
            self.avisar("desgrudou do rosto")

    def passoDesenho(self, ativas, agora):
        punho = next((m for m in ativas if m.punho), None)
        pincando = [m for m in ativas if m.pinca]
        rock = next((m for m in ativas if m.rock), None)
        self.destaque = None
        modo = self.gesto_d

        if punho is not None:
            if modo != "apagando":
                self.terminarTraco()
            modo = "apagando"
            raio = max(28.0, 0.55 * punho.palma)
            self.gesto = {"centro": punho.centro, "raio": raio}
            self.desenhos = apagar(self.desenhos, punho.centro, raio)

        elif len(pincando) >= 2:
            a, b = pincando[0], pincando[1]
            if modo != "transformando":
                self.terminarTraco(descartar=agora - self.inicio_atual < DESCARTA_TRACO)
                modo = "transformando"
                meio0 = (a.ponto + b.ponto) / 2
                alvo = (selecionar(self.desenhos, meio0) or selecionar(self.desenhos, a.ponto)
                        or selecionar(self.desenhos, b.ponto))
                if alvo is not None:
                    alvo.local = None   # solto do rosto enquanto mexe
                self.gesto = {"alvo": alvo, "M0": alvo.M.copy() if alvo else None, "meio0": meio0,
                              "dist0": max(np.linalg.norm(a.ponto - b.ponto), 1.0)}
            a.travada = b.travada = True
            alvo = self.gesto["alvo"]
            if alvo in self.desenhos:
                meio = (a.ponto + b.ponto) / 2
                s = min(max(np.linalg.norm(a.ponto - b.ponto) / self.gesto["dist0"], 0.1), 10)
                alvo.M = translacao(meio) @ escala(s) @ translacao(-self.gesto["meio0"]) @ self.gesto["M0"]
                self.destaque = alvo

        elif rock is not None:
            if modo != "girando":
                self.terminarTraco()
                modo = "girando"
                alvo = selecionar(self.desenhos, rock.centro) or selecionar(self.desenhos, rock.ponta)
                if alvo is not None:
                    alvo.local = None
                self.gesto = {"alvo": alvo, "M0": alvo.M.copy() if alvo else None, "ultimo": rock.angulo,
                              "total": 0.0, "centro": alvo.centro() if alvo else rock.centro}
            self.gesto["total"] += angulo_limpo(rock.angulo - self.gesto["ultimo"])
            self.gesto["ultimo"] = rock.angulo
            alvo = self.gesto["alvo"]
            if alvo in self.desenhos:
                c = self.gesto["centro"]
                alvo.M = translacao(c) @ rotacao(self.gesto["total"]) @ translacao(-c) @ self.gesto["M0"]
                self.destaque = alvo

        elif len(pincando) == 1 and not pincando[0].travada:
            m = pincando[0]
            if self.naBarra(m.ponto_inicio):
                modo = "espessura"
                self.esp = self.espessuraEm(m.ponto[1])
            elif m.ponto_inicio[0] < 175 or any(b.contem(m.ponto_inicio) for b in self.botoes()):
                modo = "livre"
            elif agora - m.inicio_pinca >= ESPERA_DESENHO:
                if modo != "desenhando":
                    modo = "desenhando"
                    cor = self.cores[self.cor_idx]
                    self.atual = [([], cor, self.esp)] + ([([], cor, self.esp)] if self.espelho else [])
                    self.inicio_atual = agora
                p = m.ponto
                for i, (pts, _, _) in enumerate(self.atual):
                    q = p if i == 0 else np.array([self.w - p[0], p[1]])
                    if not pts or np.linalg.norm(np.array(pts[-1]) - q) >= 2:
                        pts.append(q.tolist())
        else:
            if modo == "desenhando":
                self.terminarTraco()
            elif modo in ("transformando", "girando"):
                alvo = self.gesto.get("alvo")
                if alvo in self.desenhos:
                    self.grudarSePrecisar(alvo)
            modo = "livre"

        if modo != self.gesto_d and self.gesto_d in ("transformando", "girando") and modo != "livre":
            alvo = self.gesto.get("alvo") if isinstance(self.gesto, dict) else None
            if alvo in self.desenhos:
                self.grudarSePrecisar(alvo)
        self.gesto_d = modo

        # apontar pra uma cor
        mirando = None
        for m in ativas:
            if m.apontando:
                i = self.corEm(m.ponta)
                if i is not None:
                    mirando = i
                    break
        self.progresso_cor = 0.0
        if mirando is None or mirando != self.mira_cor["idx"]:
            self.mira_cor = {"idx": mirando, "desde": agora}
        else:
            self.progresso_cor = min((agora - self.mira_cor["desde"]) / ESPERA_COR, 1.0)
            if self.progresso_cor >= 1.0:
                self.cor_idx = mirando

    def corEm(self, p):
        for i, c in enumerate(self.pos_cores):
            if np.linalg.norm(np.array(c) - p) <= 30:
                return i
        return None

    def naBarra(self, p):
        x, y0, y1 = self.barra
        return abs(p[0] - x) < 30 and y0 - 24 <= p[1] <= y1 + 24

    def espessuraEm(self, y):
        x, y0, y1 = self.barra
        f = (y1 - y) / (y1 - y0)
        return int(round(ESP_MIN + min(max(f, 0), 1) * (ESP_MAX - ESP_MIN)))

    # ------------------------------------------------------------ fisica
    def entrarFisica(self):
        for d in self.desenhos:
            if d.local is not None and self.rosto_on:
                continue   # grudado no rosto: fica so de enfeite
            d.local = None
            c = CorpoDesenho(self.espaco, d)
            d.corpo = c
            self.corpos.append(c)
            for s in c.shapes:
                self.dono[s] = c

    def sairFisica(self):
        self.soltarTudo()
        for c in self.corpos:
            c.atualizarPose()
            for s in c.shapes:
                self.dono.pop(s, None)
            self.espaco.remove(c.body, *c.shapes)
            c.d.corpo = None
        self.corpos = []
        self.tirarCabeca()

    def fisicos(self):
        return self.coisas + self.corpos

    def removerCoisa(self, o):
        self.tirarRestricao(o)
        for k in o.maos:
            if k in self.fisicas and self.fisicas[k].segurando is o:
                self.fisicas[k].segurando = None
        self.dono.pop(o.shape, None)
        self.espaco.remove(o.body, o.shape)
        self.coisas.remove(o)

    def tirarRestricao(self, o):
        if o.restricao is not None:
            self.espaco.remove(o.restricao)
            o.restricao = None

    def pegar(self, o, chave, ponto):
        self.tirarRestricao(o)
        if o.body.body_type != pymunk.Body.DYNAMIC:
            o.body.body_type = pymunk.Body.DYNAMIC
        f = self.fisicas[chave]
        ancora = o.body.world_to_local(tuple(ponto))
        if isinstance(o, Coisa) and o.t["forma"] == "bola" and np.linalg.norm(np.array(ponto) - o.pos) > o.raio:
            ancora = (0, 0)
        junta = pymunk.PivotJoint(f.body, o.body, (0, 0), ancora)
        junta.max_force = 150000 * o.massa
        junta.error_bias = pow(1 - 0.5, 60)
        self.espaco.add(junta)
        o.restricao, o.estado, o.maos = junta, "pego", [chave]
        o.esticar, o.balanco = (1.0, 0.0), None
        f.segurando = o

    def soltarLivre(self, o, velocidade=None):
        self.tirarRestricao(o)
        if o.body.body_type != pymunk.Body.DYNAMIC:
            o.body.body_type = pymunk.Body.DYNAMIC
        if velocidade is not None:
            o.body.velocity = tuple(velocidade)
        for chave in o.maos:
            if chave in self.fisicas and self.fisicas[chave].segurando is o:
                self.fisicas[chave].segurando = None
        o.estado, o.maos = "livre", []
        if isinstance(o, Coisa) and o.t.get("bomba") and o.pavio is None:
            o.pavio = time.time() + BOMBA_PAVIO

    def soltarTudo(self):
        for f in self.fisicas.values():
            if f.segurando is not None and f.segurando in self.fisicos():
                self.soltarLivre(f.segurando)
            f.segurando, f.pinca_antes, f.aberta_desde = None, False, None

    def comecarEsticar(self, o, a, b, ativas):
        self.tirarRestricao(o)
        o.body.body_type = pymunk.Body.KINEMATIC
        pa, pb = self.pontoDe(ativas[a]), self.pontoDe(ativas[b])
        eixo = pb - pa
        o.gesto = {"dist0": max(np.linalg.norm(eixo), 30.0), "ang0": math.atan2(eixo[1], eixo[0]),
                   "angulo_obj0": o.body.angle, "meios": deque(maxlen=6)}
        o.estado, o.maos, o.balanco = "esticado", [a, b], None
        self.fisicas[a].segurando = self.fisicas[b].segurando = o

    def objetoEm(self, p, chave):
        info = self.espaco.point_query_nearest(tuple(p), ALCANCE, PEGAVEIS)
        if info is not None and info.shape is not None and info.shape in self.dono:
            return self.dono[info.shape]
        perto = [(np.linalg.norm(o.pos - p), o) for o in self.coisas
                 if o.estado == "pego" and o.maos and o.maos[0] != chave]
        perto = [(d, o) for d, o in perto if d < o.raio + ALCANCE_2A_MAO]
        return min(perto, key=lambda x: x[0])[1] if perto else None

    def segura(self, m):
        return m.pinca if self.pegar_com == "pinca" else m.punho

    def pontoDe(self, m):
        return m.ponto if self.pegar_com == "pinca" else m.centro

    def explodir(self, b):
        for k in list(b.maos):
            if k in self.fisicas:
                self.fisicas[k].segurando, self.fisicas[k].pinca_antes = None, False
        centro = b.pos
        self.removerCoisa(b)
        self.explosoes.append((centro, time.time()))
        self.faiscas(centro, (60, 170, 255), 40, 900)
        for o in self.fisicos():
            d = o.pos - centro
            dist = np.linalg.norm(d)
            if dist > BOMBA_RAIO or dist < 1e-3:
                continue
            if o.estado != "livre" or o.body.body_type != pymunk.Body.DYNAMIC:
                self.soltarLivre(o)
            o.body.velocity = tuple(np.array(o.body.velocity) + d / dist * BOMBA_FORCA * (1 - dist / BOMBA_RAIO)
                                    / max(o.massa, 1.0) ** 0.5)
            o.body.angular_velocity += 10 * (1 if d[0] > 0 else -1)

    def faiscas(self, centro, cor, n, vel):
        rng = np.random.default_rng()
        agora = time.time()
        for _ in range(n):
            a = rng.uniform(0, 2 * math.pi)
            v = np.array([math.cos(a), math.sin(a)]) * rng.uniform(0.2, 1) * vel
            self.particulas.append([np.array(centro, float), v, cor, agora, rng.uniform(0.3, 0.7), rng.uniform(3, 7)])

    def tirarCabeca(self):
        if self.cabeca is not None:
            self.espaco.remove(*self.cabeca)
            self.cabeca = None

    def passoFisica(self, maos, ativas, agora, dt):
        esp = self.espaco
        # maos -> corpos
        for chave in ativas:
            if chave not in self.fisicas:
                self.fisicas[chave] = MaoFisica(esp)
        for chave in [k for k in self.fisicas if k not in maos]:
            f = self.fisicas.pop(chave)
            if f.segurando is not None and f.segurando in self.fisicos():
                self.soltarLivre(f.segurando, f.velocidade())
            esp.remove(f.body, f.shape)
        for k, f in self.fisicas.items():
            if k in ativas:
                f.mover(self.pontoDe(ativas[k]), agora, dt)

        # soltar
        soltaram = []
        for k, f in self.fisicas.items():
            if not f.pinca_antes:
                continue
            if k in ativas and self.segura(ativas[k]):
                f.aberta_desde = f.perdida_desde = None
            elif k in ativas:
                f.perdida_desde = None
                f.aberta_desde = f.aberta_desde or agora
                if f.segurando is None or agora - f.aberta_desde >= SOLTAR_ESPERA:
                    soltaram.append(k)
            else:
                f.perdida_desde = f.perdida_desde or agora
                if f.segurando is None or agora - f.perdida_desde >= PERDIDA_MAX:
                    soltaram.append(k)
        for k in soltaram:
            f = self.fisicas[k]
            f.pinca_antes, f.solto_em = False, agora
            f.aberta_desde = f.perdida_desde = None
            o = f.segurando
            if o is None or o not in self.fisicos():
                f.segurando = None
                continue
            if o.estado == "pego":
                self.soltarLivre(o, f.velocidade())
            elif o.estado == "esticado":
                outra = next(c for c in o.maos if c != k)
                outra_soltou = outra in soltaram or (outra in self.fisicas and self.fisicas[outra].aberta_desde is not None)
                s, eixo = o.esticar
                meios = o.gesto["meios"]
                v_meio = ((meios[-1][1] - meios[0][1]) / max(meios[-1][0] - meios[0][0], 1e-3)
                          if len(meios) >= 2 else np.zeros(2))
                if outra_soltou or outra not in self.fisicas:
                    self.soltarLivre(o, v_meio)
                    o.balanco = (s, eixo, agora)
                    if outra in self.fisicas:
                        self.fisicas[outra].pinca_antes = False
                else:
                    o.body.body_type = pymunk.Body.DYNAMIC
                    o.body.velocity = tuple(v_meio)
                    mola = pymunk.DampedSpring(self.fisicas[outra].body, o.body, (0, 0), (0, 0), 0,
                                               260 * o.massa, 4 * o.massa)
                    esp.add(mola)
                    o.restricao, o.estado, o.maos = mola, "elastico", [outra]
                    f.segurando = None
            elif o.estado == "elastico":
                s, eixo = o.esticar
                self.soltarLivre(o)
                o.balanco = (s, eixo, agora)

        # pegar
        for k, m in ativas.items():
            f = self.fisicas[k]
            if not self.segura(m) or f.pinca_antes:
                continue
            f.pinca_antes = True
            p = self.pontoDe(m)
            tipo = self.itemEm(p)
            if tipo is not None:
                if len(self.coisas) >= MAX_OBJETOS:
                    self.removerCoisa(self.coisas[0])
                novo = Coisa(esp, tipo, p)
                self.coisas.append(novo)
                self.dono[novo.shape] = novo
                self.pegar(novo, k, p)
                continue
            o = self.objetoEm(p, k)
            if o is None:
                continue
            if (isinstance(o, Coisa) and o.estado == "pego" and o.maos and o.maos[0] != k
                    and o.maos[0] in ativas):
                self.comecarEsticar(o, o.maos[0], k, ativas)
            elif o.estado in ("livre", "elastico", "pego"):
                self.soltarLivre(o)
                self.pegar(o, k, p)

        # esticar / balancar
        for o in self.coisas:
            if o.estado == "esticado":
                a, b = o.maos
                if a not in ativas or b not in ativas:
                    continue
                pa, pb = self.pontoDe(ativas[a]), self.pontoDe(ativas[b])
                eixo = pb - pa
                meio = (pa + pb) / 2
                ang = math.atan2(eixo[1], eixo[0])
                s = min(max(np.linalg.norm(eixo) / o.gesto["dist0"], 0.4), 3.0)
                o.body.velocity = tuple((meio - o.pos) / dt)
                alvo_ang = o.gesto["angulo_obj0"] + (ang - o.gesto["ang0"])
                o.body.angular_velocity = math.remainder(alvo_ang - o.body.angle, 2 * math.pi) / dt
                o.esticar = (s, ang)
                o.gesto["meios"].append((agora, meio))
            elif o.estado == "elastico":
                k = o.maos[0]
                if k in self.fisicas:
                    d = np.array(self.fisicas[k].body.position) - o.pos
                    o.esticar = (1 + min(np.linalg.norm(d) / (2 * o.raio), 2.0), math.atan2(d[1], d[0]))
            elif o.balanco is not None:
                s0, eixo, t0 = o.balanco
                t = agora - t0
                if t > 1.2:
                    o.esticar, o.balanco = (1.0, 0.0), None
                else:
                    amp = min(abs(s0 - 1), 0.5) * (1 if s0 >= 1 else -1)
                    o.esticar = (1 + amp * math.exp(-5 * t) * math.cos(22 * t), eixo)

        # tapa
        for k, f in self.fisicas.items():
            m = ativas.get(k)
            rapida = np.linalg.norm(np.array(f.body.velocity)) > TAPA_VEL
            ligado = (m is not None and not f.pinca_antes and f.segurando is None and rapida
                      and not m.punho and not m.pinca and agora - f.solto_em > 0.3)
            f.shape.filter = pymunk.ShapeFilter(categories=CAT_MAO, mask=(CAT_OBJETO | CAT_DESENHO) if ligado else 0)

        # ima
        self.imas = [m.centro for k, m in ativas.items() if m.rock and self.fisicas[k].segurando is None]
        for c in self.imas:
            for o in self.fisicos():
                if o.estado != "livre" or o.body.body_type != pymunk.Body.DYNAMIC:
                    continue
                d = c - o.pos
                dist = np.linalg.norm(d)
                if 1 < dist < IMA_RAIO:
                    v = np.array(o.body.velocity) * (0.9 if dist < 120 else 1.0)
                    o.body.velocity = tuple(v + d / dist * IMA_FORCA * dt)

        # bombas
        for o in [o for o in self.coisas if o.pavio is not None and agora >= o.pavio]:
            self.explodir(o)

        # rosto: cabeca rebate, boca come
        if self.rosto_on and self.rosto.visivel:
            r = self.rosto
            alvo = r.olhos + np.array([math.sin(r.angulo), -math.cos(r.angulo)]) * r.dist_olhos * 0.35
            raio = r.dist_olhos * 0.95
            if self.cabeca is None:
                corpo = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                corpo.position = tuple(alvo)
                forma = pymunk.Circle(corpo, raio)
                forma.elasticity, forma.friction = 0.8, 0.6
                forma.filter = pymunk.ShapeFilter(categories=CAT_ROSTO)
                esp.add(corpo, forma)
                self.cabeca = (corpo, forma)
            corpo, forma = self.cabeca
            corpo.velocity = tuple((alvo - np.array(corpo.position)) / dt)
            forma.unsafe_set_radius(raio)
            if r.aberta:
                for o in list(self.coisas):
                    if o.estado == "livre" and np.linalg.norm(o.pos - r.boca) < o.raio + r.boca_larg * 0.55:
                        if o.t.get("bomba"):
                            self.explodir(o)
                            self.avisar("BOOM na boca!")
                        else:
                            self.faiscas(r.boca, o.t["cor"], 18, 350)
                            self.removerCoisa(o)
                            self.comidos += 1
                            self.avisar(f"nham! ({self.comidos})")
        else:
            self.tirarCabeca()

        # passo da fisica
        esp.gravity = (0, GRAVIDADE * MODOS_GRAVIDADE[self.modo_g][1])
        for _ in range(SUBPASSOS):
            esp.step(dt / SUBPASSOS)
        w, h = self.w, self.h
        for o in self.fisicos():
            if o.body.body_type != pymunk.Body.DYNAMIC:
                continue
            v = np.array(o.body.velocity)
            n = np.linalg.norm(v)
            if n > VEL_MAX:
                v = v * VEL_MAX / n
            r = o.raio if isinstance(o, Coisa) else 10
            x, y = o.body.position
            q = o.t["quique"]
            if x < r or x > w - r:
                x, v[0] = min(max(x, r), w - r), -v[0] * q
            if y < r or y > h - r:
                y, v[1] = min(max(y, r), h - r), -v[1] * q
            if (x, y) != tuple(o.body.position):
                o.body.position = (x, y)
            o.body.velocity = tuple(v)
            if np.linalg.norm(v) > 350:
                o.rastro.append(tuple(o.pos.astype(int)))
            elif o.rastro:
                o.rastro.popleft()
        for c in self.corpos:
            c.atualizarPose()

    def itemEm(self, p):
        if not self.aba_aberta:
            return None
        for tipo, c in self.itens:
            if np.linalg.norm(np.array(c) - p) < self.raio_item:
                return tipo
        return None

    # ------------------------------------------------------------ tela
    def desenharCoisa(self, img, o, agora):
        spr, luz, luz_gira = self.sprites[o.tipo]
        lado = spr.shape[0]
        c = (lado / 2, lado / 2)
        s, eixo = o.esticar
        S = esticamento(s, eixo)
        a = o.body.angle
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        A = afim(S @ R, o.pos, c)
        colar(img, spr, A, luz, A if luz_gira else afim(S, o.pos, c))
        r = o.raio
        p = o.pos
        if o.tipo == "balao":   # no e barbante (sempre pra baixo)
            base = p + [0, r * 0.98]
            cv.fillPoly(img, [np.array([base + [-5, 6], base + [5, 6], base + [0, -2]]).astype(np.int32)],
                        (50, 40, 190), cv.LINE_AA)
            fio = [base + [math.sin(agora * 3 + k * 0.6) * 4, 6 + k * r * 0.12] for k in range(12)]
            cv.polylines(img, [np.array(fio).astype(np.int32)], False, (220, 220, 220), 1, cv.LINE_AA)
        elif o.tipo == "bomba":
            topo = p + R @ np.array([0, -1.0 * r])
            ponta = topo + R @ np.array([r * 0.35, -r * 0.45])
            meio = topo + R @ np.array([r * 0.05, -r * 0.35])
            cv.polylines(img, [np.array([topo, meio, ponta]).astype(np.int32)], False, (90, 170, 220), 3, cv.LINE_AA)
            if o.pavio is not None:
                falta = max(o.pavio - agora, 0)
                brilho = int(6 + 4 * math.sin(agora * 40))
                cv.circle(img, tuple(ponta.astype(int)), brilho + 4, (40, 150, 255), -1, cv.LINE_AA)
                cv.circle(img, tuple(ponta.astype(int)), brilho, (200, 245, 255), -1, cv.LINE_AA)
                pulso = 0.5 + 0.5 * math.sin(agora * (8 + 30 * (1 - falta / BOMBA_PAVIO)))
                cv.circle(img, tuple(p.astype(int)), int(r + 6 + 6 * pulso), (60, 60, 255), 2, cv.LINE_AA)
                self.escritor.escrever(img, f"{falta:.1f}", (p[0], p[1] - r - 26), 18, (120, 210, 255), True, True)

    def desenharTela(self, frame, ativas, agora):
        es = self.escritor
        w, h = self.w, self.h
        # rastros e objetos
        saida = frame
        if self.rastro:
            for o in self.coisas:
                pts = list(o.rastro)
                for i in range(1, len(pts)):
                    cv.line(saida, pts[i - 1], pts[i], o.t["cor"], max(1, int(o.raio * 0.5 * i / len(pts))), cv.LINE_AA)
        lista = list(self.desenhos)
        if self.atual:
            lista.append(Desenho([Traco(pts, cor, e) for pts, cor, e in self.atual if len(pts) >= 2]))
        saida = desenharDesenhos(saida, lista, agora, getattr(self, "destaque", None) if self.modo == "desenho" else None)
        for o in self.coisas:
            self.desenharCoisa(saida, o, agora)
        for o in self.coisas:
            if o.estado == "elastico" and o.maos and o.maos[0] in self.fisicas:
                q = tuple(int(v) for v in self.fisicas[o.maos[0]].body.position)
                cv.line(saida, q, tuple(o.pos.astype(int)), (255, 255, 255), 2, cv.LINE_AA)

        # efeitos
        for pos, t0 in list(self.explosoes):
            t = agora - t0
            if t > 0.4:
                self.explosoes.remove((pos, t0))
                continue
            c = tuple(int(v) for v in pos)
            cv.circle(saida, c, int(40 + BOMBA_RAIO * t / 0.4), (40, 160, 255), max(2, int(14 * (1 - t / 0.4))), cv.LINE_AA)
            cv.circle(saida, c, int(70 * (1 - t / 0.4)) + 1, (200, 240, 255), -1, cv.LINE_AA)
        vivas = []
        for p in self.particulas:
            pos, v, cor, t0, vida, tam = p
            t = agora - t0
            if t > vida:
                continue
            dt = 1 / 30
            p[0] = pos + v * dt
            p[1] = v * 0.94 + np.array([0, 900]) * dt
            cv.circle(saida, tuple(p[0].astype(int)), max(1, int(tam * (1 - t / vida))), cor, -1, cv.LINE_AA)
            vivas.append(p)
        self.particulas = vivas
        for c in getattr(self, "imas", []) if self.modo == "fisica" else []:
            fase = (agora * 1.5) % 1
            for k in range(3):
                raio = int(IMA_RAIO * 0.6 * ((fase + k / 3) % 1))
                cv.circle(saida, tuple(int(v) for v in c), max(raio, 1), COR_ROSTO, 1, cv.LINE_AA)

        # rosto
        if self.rosto_on and self.rosto.visivel:
            r = self.rosto
            movendo = self.modo == "desenho" and self.gesto_d in ("transformando", "girando") and \
                self.gesto.get("alvo") is not None and r.contem(self.gesto["alvo"].centro())
            if movendo:   # vai grudar: mostra o contorno
                oval = ((r.oval - r.centro_oval) * 1.5 + r.centro_oval).astype(np.int32)
                cv.polylines(saida, [oval], True, COR_ROSTO, 2, cv.LINE_AA)
                es.escrever(saida, "solte pra grudar no rosto", (r.centro_oval[0], r.oval[:, 1].min() - 40), 20,
                            COR_ROSTO, True, True)
            if self.modo == "fisica":
                boca = tuple(r.boca.astype(int))
                cor = (80, 255, 160) if r.aberta else (200, 200, 200)
                cv.ellipse(saida, boca, (int(r.boca_larg * 0.6), int(r.boca_larg * (0.45 if r.aberta else 0.18))),
                           math.degrees(r.angulo), 0, 360, cor, 2, cv.LINE_AA)

        self.desenharLateral(saida, ativas, agora)
        self.desenharBotoes(saida, agora)
        self.desenharMaos(saida, ativas)

        # status e ajuda
        x0 = 190
        acento = self.corAcento(agora)
        info = (f"{self.fps:.0f} FPS   {len(self.desenhos)} desenho(s)" if self.modo == "desenho" else
                f"{self.fps:.0f} FPS   {len(self.coisas)} objeto(s)   comidos {self.comidos}"
                + ("   camera lenta" if self.lenta else ""))
        es.escrever(saida, info, (x0, h - 34 if not self.ajuda else h - 54 - 26 * len(self.textoAjuda())), 15, APAGADO)
        nomes = {"desenhando": "desenhando", "girando": "girando", "transformando": "movendo / redimensionando",
                 "apagando": "apagando", "espessura": f"espessura {self.esp}"}
        if self.modo == "desenho" and self.gesto_d in nomes:
            txt = nomes[self.gesto_d]
            if self.gesto_d in ("girando", "transformando") and self.gesto.get("alvo") is None:
                txt = "nenhum desenho embaixo da mao"
            es.escrever(saida, txt, (x0, 76), 24, acento, True)
        if self.aviso and agora - self.aviso[1] < 1.6:
            msg = self.aviso[0]
            lw = es.largura(msg, 22, True)
            painel(saida, w / 2 - lw / 2 - 18, 74, w / 2 + lw / 2 + 18, 116, 20, alfa=0.7)
            es.escrever(saida, msg, (w / 2, 95), 22, acento, True, True)
        if self.ajuda:
            linhas = self.textoAjuda()
            y = h - 16 - 26 * len(linhas)
            largura = max(es.largura(l, 15) for l in linhas)
            painel(saida, x0 - 12, y - 8, x0 + largura + 14, h - 10, 12, alfa=0.5)
            for i, l in enumerate(linhas):
                es.escrever(saida, l, (x0, y + 26 * i), 15, TEXTO if i < len(linhas) - 1 else APAGADO)
        return saida

    def corAcento(self, agora):
        if self.modo == "fisica":
            return COR_FISICA
        c = self.cores[self.cor_idx]
        return corArco(0, agora) if c is ARCO else c

    def textoAjuda(self):
        if self.modo == "desenho":
            return ["pinca: desenha   |   punho: apaga   |   rock + girar: gira",
                    "pinca com as 2 maos: move e redimensiona" + ("   |   solte no rosto pra grudar" if self.rosto_on else ""),
                    "TAB modo  O rosto  E espelho  Z desfaz  C limpa  K foto  F tela cheia  H ajuda  Q sai"]
        return [f"{self.pegar_com} na aba: cria   |   no objeto ou desenho: pega   |   soltar mexendo: joga",
                "2 maos: estica / estilingue   |   mao aberta rapida: tapa   |   rock: ima"
                + ("   |   boca aberta: come" if self.rosto_on else ""),
                "TAB modo  O rosto  P segurar  G gravidade  L lenta  R rastro  X fixa  Z  C  K  H  Q"]

    def desenharLateral(self, img, ativas, agora):
        es = self.escritor
        h = self.h
        if self.modo == "desenho":
            painel(img, 12, 84, 168, self.barra[2] + 96, 18)
            es.escrever(img, "cores", (26, 94), 16, APAGADO, True)
            for i, (c, centro) in enumerate(zip(self.cores, self.pos_cores)):
                if c is ARCO:
                    for k in range(12):
                        cv.ellipse(img, centro, (22, 22), 0, k * 30, k * 30 + 31, corArco(k * 10, agora), -1, cv.LINE_AA)
                else:
                    cv.circle(img, centro, 22, c, -1, cv.LINE_AA)
                cv.circle(img, centro, 22, (255, 255, 255), 1, cv.LINE_AA)
                if i == self.cor_idx:
                    cv.circle(img, centro, 28, (255, 255, 255), 2, cv.LINE_AA)
                if i == self.mira_cor["idx"] and getattr(self, "progresso_cor", 0) > 0:
                    cv.ellipse(img, centro, (32, 32), -90, 0, 360 * self.progresso_cor, (255, 255, 255), 3, cv.LINE_AA)
            x, y0, y1 = self.barra
            cv.line(img, (x, y0), (x, y1), (80, 80, 80), 8, cv.LINE_AA)
            y = int(y1 - (self.esp - ESP_MIN) / (ESP_MAX - ESP_MIN) * (y1 - y0))
            acento = self.corAcento(agora)
            cv.line(img, (x, y), (x, y1), acento, 8, cv.LINE_AA)
            cv.circle(img, (x, y), 13 if self.gesto_d == "espessura" else 10, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(img, (90, y1 + 50), max(2, self.esp // 2), acento, -1, cv.LINE_AA)
            es.escrever(img, f"{self.esp}px", (x, y0 - 22), 14, APAGADO, False, True)
        elif self.aba_aberta:
            painel(img, 12, 84, 152, self.itens[-1][1][1] + self.raio_item + 26, 18)
            es.escrever(img, "objetos", (26, 94), 16, APAGADO, True)
            for tipo, (cx, cy) in self.itens:
                spr, luz, gira = self.sprites[tipo]
                lado = spr.shape[0]
                k = min((self.raio_item - 14) / TIPOS[tipo]["tamanho"], 1.2)
                L = np.eye(2) * k
                A = afim(L, (cx, cy - 6), (lado / 2, lado / 2))
                colar(img, spr, A, luz, A, sombra=False)
                es.escrever(img, TIPOS[tipo]["nome"], (cx, cy + self.raio_item - 12), 13, APAGADO, False, True)

    def desenharBotoes(self, img, agora):
        es = self.escritor
        for b in self.botoes():
            x0, y0, x1, y1 = b.rect
            ativo = (b.nome == self.modo or (b.nome == "rosto" and self.rosto_on) or
                     (b.nome == "espelho" and self.espelho))
            cor = b.cor or (COR_FISICA if b.nome == "fisica" else self.corAcento(agora))
            painel(img, x0, y0, x1, y1, (y1 - y0) // 2, cor=(cor if ativo else VIDRO), alfa=0.85 if ativo else 0.6)
            es.escrever(img, b.rotulo(), ((x0 + x1) / 2, (y0 + y1) / 2), 17,
                        (20, 20, 20) if ativo else TEXTO, True, True, sombra=not ativo)
            if self.mira_botao["nome"] == b.nome:
                prog = min((agora - self.mira_botao["desde"]) / ESPERA_BOTAO, 1.0)
                if prog > 0:
                    cv.line(img, (x0 + 14, y1 + 5), (int(x0 + 14 + (x1 - x0 - 28) * prog), y1 + 5),
                            (255, 255, 255), 3, cv.LINE_AA)

    def desenharMaos(self, img, ativas):
        acento = self.corAcento(time.time())
        for k, m in ativas.items():
            if self.modo == "fisica":
                f = self.fisicas.get(k)
                p = tuple(int(v) for v in self.pontoDe(m))
                cor = COR_FISICA if f is not None and f.segurando is not None else (255, 255, 255)
                if self.segura(m):
                    cv.circle(img, p, 12 if m.pinca else 20, cor, -1, cv.LINE_AA)
                    cv.circle(img, p, 12 if m.pinca else 20, (0, 0, 0), 1, cv.LINE_AA)
                else:
                    cv.circle(img, p, 24, (0, 0, 0), 4, cv.LINE_AA)
                    cv.circle(img, p, 24, cor, 2, cv.LINE_AA)
                if f is not None and f.shape.filter.mask:
                    cv.circle(img, tuple(int(v) for v in f.body.position), 42, COR_FISICA, 3, cv.LINE_AA)
            else:
                if m.punho:
                    if self.gesto_d == "apagando":
                        c = tuple(int(v) for v in self.gesto["centro"])
                        cv.circle(img, c, int(self.gesto["raio"]), (0, 0, 0), 4, cv.LINE_AA)
                        cv.circle(img, c, int(self.gesto["raio"]), (255, 255, 255), 2, cv.LINE_AA)
                    continue
                p = tuple(int(v) for v in m.ponto)
                if m.pinca:
                    cv.circle(img, p, max(5, self.esp // 2 + 3), acento, -1, cv.LINE_AA)
                else:
                    cv.circle(img, p, 12, (0, 0, 0), 4, cv.LINE_AA)
                    cv.circle(img, p, 12, acento, 2, cv.LINE_AA)
            if m.apontando:
                cv.circle(img, tuple(int(v) for v in m.ponta), 7, (255, 255, 255), -1, cv.LINE_AA)
        if self.modo == "desenho" and self.gesto_d == "girando" and self.gesto.get("alvo") in self.desenhos:
            c = tuple(int(v) for v in self.gesto["centro"])
            cv.circle(img, c, 6, (255, 255, 255), -1, cv.LINE_AA)
            cv.ellipse(img, c, (40, 40), 0, 0, math.degrees(self.gesto["total"]) % 360 or 1, acento, 3, cv.LINE_AA)

    def verBotoes(self, ativas, agora):
        mirado = None
        for m in ativas.values():
            if m.apontando:
                mirado = next((b for b in self.botoes() if b.contem(m.ponta)), None)
                if mirado:
                    break
        nome = mirado.nome if mirado else None
        if nome != self.mira_botao["nome"]:
            self.mira_botao = {"nome": nome, "desde": agora}
        elif mirado is not None and agora - self.mira_botao["desde"] >= ESPERA_BOTAO:
            mirado.acao()
            self.mira_botao = {"nome": nome, "desde": agora + 10}   # so de novo depois de tirar o dedo

    # ------------------------------------------------------------ teclado
    def tecla(self, t):
        if t in (9, ord("m")):
            self.trocarModo()
        elif t == ord("o"):
            self.trocarRosto()
        elif t == ord("z"):
            self.desfazer()
        elif t == ord("c"):
            self.limpar()
        elif t == ord("h"):
            self.ajuda = not self.ajuda
        elif self.modo == "desenho" and t == ord("e"):
            self.trocarEspelho()
        elif self.modo == "fisica":
            acoes = {ord("p"): self.trocarPegar, ord("g"): self.trocarGravidade, ord("x"): self.fixarDesenhos,
                     ord("l"): lambda: setattr(self, "lenta", not self.lenta),
                     ord("r"): lambda: setattr(self, "rastro", not self.rastro),
                     ord("a"): lambda: setattr(self, "aba_aberta", not self.aba_aberta)}
            if t in acoes:
                acoes[t]()

    # ------------------------------------------------------------ loop
    def rodar(self):
        args = self.args
        baixarModelo()
        mp, detector = criarDetector(num_maos=2)
        try:
            captura, _ = abrirFonte(args.fonte)
        except OSError as e:
            raise SystemExit(f"nao conectou na fonte {args.fonte}: {e}")
        if isinstance(captura, cv.VideoCapture):
            captura.set(cv.CAP_PROP_FRAME_WIDTH, args.largura)
            captura.set(cv.CAP_PROP_FRAME_HEIGHT, args.altura)
            captura.set(cv.CAP_PROP_FPS, 60)
        cv.namedWindow(JANELA, cv.WINDOW_NORMAL)
        cv.resizeWindow(JANELA, args.largura, args.altura)
        cheia = args.cheia
        if cheia:
            cv.setWindowProperty(JANELA, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        maos = {}
        ultimo_ts = -1
        t_anterior = time.time()
        self.fps = 0.0
        try:
            while True:
                ok, frame = captura.read()
                if not ok:
                    break
                if not args.sem_espelho:
                    frame = cv.flip(frame, 1)
                h, w = frame.shape[:2]
                if w != args.largura:
                    frame = cv.resize(frame, (args.largura, round(h * args.largura / w)))
                    h, w = frame.shape[:2]
                if self.w is None:
                    self.montar(w, h)
                agora = time.time()
                dt_real = min(max(agora - t_anterior, 1 / 120), 1 / 15)
                t_anterior = agora
                dt = dt_real * (0.3 if self.lenta and self.modo == "fisica" else 1.0)

                rgb = np.ascontiguousarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                ultimo_ts = max(int(time.monotonic() * 1000), ultimo_ts + 1)
                resultado = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ultimo_ts)
                associarMaos(resultado, maos, w, h, agora)
                for chave in [k for k, m in maos.items() if agora - m.vista > 1.0]:
                    del maos[chave]
                ativas = {k: m for k, m in maos.items() if agora - m.vista <= SUMIU}

                if self.rosto_on:
                    self.rosto.atualizar(mp, rgb, w, h, agora)
                    if self.rosto.visivel:   # desenhos grudados seguem o rosto
                        for d in self.desenhos:
                            if d.local is not None and d.corpo is None:
                                d.M = self.rosto.F @ d.local

                self.verBotoes(ativas, agora)
                if self.modo == "desenho":
                    self.passoDesenho(list(ativas.values()), agora)
                else:
                    self.passoFisica(maos, ativas, agora, dt)

                saida = self.desenharTela(frame, ativas, agora)
                gasto = max(time.time() - agora, 1e-3)
                self.fps = 0.9 * self.fps + 0.1 / gasto if self.fps else 1 / gasto
                cv.imshow(JANELA, saida)

                t = cv.waitKey(1) & 0xFF
                if t in (ord("q"), 27) or cv.getWindowProperty(JANELA, cv.WND_PROP_VISIBLE) < 1:
                    break
                if t == ord("f"):
                    cheia = not cheia
                    cv.setWindowProperty(JANELA, cv.WND_PROP_FULLSCREEN,
                                         cv.WINDOW_FULLSCREEN if cheia else cv.WINDOW_NORMAL)
                elif t == ord("k"):
                    pasta = os.path.join(PASTA, "capturas")
                    os.makedirs(pasta, exist_ok=True)
                    caminho = os.path.join(pasta, time.strftime("estudio_%Y%m%d_%H%M%S.png"))
                    cv.imwrite(caminho, saida)
                    self.avisar("foto salva em gestos/capturas")
                elif t != 255:
                    self.tecla(t)
        finally:
            captura.release()
            detector.close()
            self.rosto.fechar()
            cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Desenho + fisica com as maos (e o rosto, se quiser).")
    parser.add_argument("--fonte", default="webcam", help="webcam, numero da camera, URL (DroidCam/ESP32) ou video")
    parser.add_argument("--largura", type=int, default=1280)
    parser.add_argument("--altura", type=int, default=720)
    parser.add_argument("--cheia", action="store_true")
    parser.add_argument("--sem-espelho", action="store_true")
    Estudio(parser.parse_args()).rodar()


if __name__ == "__main__":
    main()
