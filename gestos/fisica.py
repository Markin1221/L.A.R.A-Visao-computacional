"""
Caixa de areia com fisica: pegue objetos com a mao, jogue, estique e lance como estilingue.

  aba de objetos (esquerda)           -> feche a mao (ou pinca) num item da aba pra criar um na sua mao
  punho ou pinca num objeto           -> pega (segura pelo ponto onde pegou)
  abrir a mao mexendo                 -> joga (a velocidade da mao vai pro objeto)
  fechar a outra mao no objeto        -> pega no ar / passa de uma mao pra outra
  as DUAS maos fechadas no objeto     -> estica (afaste as maos)
     solta as duas juntas             -> ele volta ao formato balancando
     solta UMA e depois a OUTRA       -> estilingue: ele encolhe em direcao a mao que ficou
                                         e sai voando quando voce solta
  apontar o indicador pra lingueta    -> abre/fecha a aba
  mao aberta passando rapido          -> tapa (empurra os objetos)
  rock (indicador + mindinho)         -> ima: puxa os objetos soltos
  bomba                               -> explode 2.5 s depois de soltar

Teclas: A abre/fecha a aba | G gravidade (normal/lua/zero/invertida) | S camera lenta | R rastro | C limpa | Z apaga o ultimo
        F tela cheia | H ajuda | Q ou ESC sai

Uso:
    pip install mediapipe pymunk
    python gestos/fisica.py
    python gestos/fisica.py --fonte 192.168.1.6
"""
import argparse
import math
import time
from collections import deque

import cv2 as cv
import numpy as np
import pymunk

from desenhar import abrirFonte, associarMaos, baixarModelo, criarDetector, SUMIU

JANELA = "L.A.R.A | fisica com as maos"
GRAVIDADE = 1800.0        # px/s^2 (pra baixo)
SUBPASSOS = 4             # passos de fisica por frame (colisoes mais estaveis)
MAX_OBJETOS = 30
VEL_MAX = 5000.0
ALCANCE = 60              # px: quao perto da borda do objeto o centro da mao precisa estar pra pegar
SOLTAR_ESPERA = 0.07      # s com a pinca aberta pra valer como soltar (um frame ruim nao derruba o objeto)
PERDIDA_MAX = 0.25        # s que o objeto segue a trajetoria da mao quando o detector perde ela (mao borrada)
ESPERA_ABA = 0.4          # s apontando pra lingueta pra abrir/fechar
ALCANCE_2A_MAO = 130      # px a mais pra segunda mao pegar um objeto que ja esta na outra (pra esticar)
TAPA_VEL = 700.0          # px/s: mao aberta mais rapida que isso empurra os objetos (tapa)
IMA_FORCA = 5200.0        # px/s^2: puxao do ima (sinal do rock)
IMA_RAIO = 520            # px
BOMBA_PAVIO = 2.5         # s depois de soltar a bomba
BOMBA_RAIO = 300          # px
BOMBA_FORCA = 2600.0      # px/s no centro da explosao
MODOS_GRAVIDADE = [("normal", 1.0), ("lua", 0.17), ("zero", 0.0), ("invertida", -1.0)]
CAT_PAREDE, CAT_OBJETO, CAT_MAO = 0b001, 0b010, 0b100   # a busca da mao ignora as paredes
SO_OBJETOS = pymunk.ShapeFilter(mask=CAT_OBJETO)

# ---------------------------------------------------------------- tipos de objeto
# cor em BGR; forma "bola" ou "caixa"; tamanho = raio ou meio lado
TIPOS = {
    "bola":    {"nome": "bola",    "forma": "bola",  "tamanho": 38, "massa": 1.0, "quique": 0.85, "atrito": 0.6,
                "cor": (40, 140, 255)},
    "caixa":   {"nome": "caixa",   "forma": "caixa", "tamanho": 36, "massa": 2.0, "quique": 0.25, "atrito": 0.8,
                "cor": (60, 120, 185)},
    "boliche": {"nome": "boliche", "forma": "bola",  "tamanho": 44, "massa": 8.0, "quique": 0.15, "atrito": 0.5,
                "cor": (90, 40, 40)},
    "balao":   {"nome": "balao",   "forma": "bola",  "tamanho": 42, "massa": 0.3, "quique": 0.6,  "atrito": 0.3,
                "cor": (80, 60, 235), "flutua": True},
    "superbola": {"nome": "superbola", "forma": "bola", "tamanho": 24, "massa": 0.6, "quique": 1.0, "atrito": 0.2,
                  "cor": (60, 220, 120)},
    "bomba":   {"nome": "bomba",   "forma": "bola",  "tamanho": 34, "massa": 2.5, "quique": 0.3,  "atrito": 0.6,
                "cor": (45, 45, 45), "bomba": True},
}


def flutuar(body, gravidade, amortecimento, dt):
    """Balao: 'gravidade' pra cima e bastante resistencia do ar."""
    pymunk.Body.update_velocity(body, (0, -320), 0.985, dt)


class Objeto:
    def __init__(self, espaco, tipo, pos):
        t = TIPOS[tipo]
        self.tipo, self.t = tipo, t
        self.body = pymunk.Body()
        self.body.position = tuple(pos)
        if t["forma"] == "bola":
            self.shape = pymunk.Circle(self.body, t["tamanho"])
        else:
            lado = t["tamanho"] * 2
            self.shape = pymunk.Poly.create_box(self.body, (lado, lado), radius=2)
        self.shape.mass = t["massa"]
        self.shape.elasticity = t["quique"]
        self.shape.friction = t["atrito"]
        self.shape.filter = pymunk.ShapeFilter(categories=CAT_OBJETO)
        if t.get("flutua"):
            self.body.velocity_func = flutuar
        espaco.add(self.body, self.shape)

        self.estado = "livre"     # livre | pego | esticado | elastico
        self.maos = []            # chaves das maos que estao segurando
        self.restricao = None     # junta/mola do pymunk
        self.esticar = (1.0, 0.0) # (fator ao longo do eixo, angulo do eixo) pra desenhar
        self.balanco = None       # (fator inicial, angulo, t0): balanca depois de esticar
        self.gesto = {}
        self.pavio = None         # bomba: hora em que explode
        self.rastro = deque(maxlen=14)

    @property
    def pos(self):
        return np.array(self.body.position)

    def contorno(self):
        """Pontos do contorno na tela (ja com o esticamento aplicado)."""
        c = self.pos
        if self.t["forma"] == "bola":
            ang = np.linspace(0, 2 * math.pi, 48, endpoint=False)
            r = self.t["tamanho"]
            pts = c + np.stack([np.cos(ang), np.sin(ang)], 1) * r
        else:
            pts = np.array([self.body.local_to_world(v) for v in self.shape.get_vertices()])
        s, a = self.esticar
        s = min(max(s, 0.4), 3.0)
        if abs(s - 1) > 1e-3:
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            M = R @ np.diag([s, 1 / math.sqrt(s)]) @ R.T
            pts = c + (pts - c) @ M.T
        return pts


# ---------------------------------------------------------------- mao no mundo da fisica

class MaoFisica:
    """Corpo cinematico que segue o ponto da pinca, pra prender objetos com juntas e molas."""

    def __init__(self, espaco):
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.shape = pymunk.Circle(self.body, 42)   # so colide quando a mao aberta passa rapido (tapa)
        self.shape.elasticity, self.shape.friction = 0.9, 0.5
        self.shape.filter = pymunk.ShapeFilter(categories=CAT_MAO, mask=0)
        espaco.add(self.body, self.shape)
        self.modo = None            # "pinca" ou "punho": com qual gesto pegou (o ponto segue esse gesto)
        self.historico = deque()
        self.pinca_antes = False
        self.segurando = None
        self.solto_em = 0.0
        self.aberta_desde = None    # quando a pinca abriu (espera SOLTAR_ESPERA pra soltar)
        self.perdida_desde = None   # quando o detector perdeu a mao

    def mover(self, ponto, agora, dt):
        alvo = np.asarray(ponto, float)
        pos = np.array(self.body.position)
        if not self.historico:
            self.body.position = tuple(alvo)
            pos = alvo
        self.body.velocity = tuple((alvo - pos) / max(dt, 1e-3))
        self.historico.append((agora, alvo))
        while self.historico and agora - self.historico[0][0] > 0.35:
            self.historico.popleft()

    def tapa(self, ligado):
        self.shape.filter = pymunk.ShapeFilter(categories=CAT_MAO, mask=CAT_OBJETO if ligado else 0)

    def velocidade(self):
        """
        Velocidade do arremesso: a MAIOR velocidade media (janelas de ~0.08 s) nos ultimos 0.35 s.
        Quando o detector percebe que a mao abriu, ela ja esta freando; a velocidade 'de agora' seria ~0
        e o objeto so cairia.
        """
        h = list(self.historico)
        if len(h) < 2:
            return np.zeros(2)
        v, melhor = np.zeros(2), 0.0
        j = 0
        for i in range(1, len(h)):
            while j < i - 1 and h[i][0] - h[j + 1][0] >= 0.08:
                j += 1
            dt = h[i][0] - h[j][0]
            if dt < 0.03:
                continue
            vi = (h[i][1] - h[j][1]) / dt
            ni = np.linalg.norm(vi)
            if ni > melhor:
                v, melhor = vi, ni
        n = melhor
        return v * (VEL_MAX / n) if n > VEL_MAX else v


# ---------------------------------------------------------------- aba de objetos

class Aba:
    def __init__(self, h):
        self.aberta = True
        self.largura = 150
        passo = min(120, (h - 110) // len(TIPOS))
        self.raio = min(52, passo // 2)
        self.itens = [(tipo, (75, int(100 + passo / 2 + i * passo))) for i, tipo in enumerate(TIPOS)]
        self.lingueta = (0, h // 2 - 50, 26, 100)   # x, y, l, a (quando fechada fica na borda)
        self.mira_desde = None

    def linguetaRect(self):
        x, y, l, a = self.lingueta
        if self.aberta:
            x = self.largura
        return x, y, l, a

    def naLingueta(self, p):
        x, y, l, a = self.linguetaRect()
        return x - 10 <= p[0] <= x + l + 10 and y <= p[1] <= y + a

    def itemEm(self, p):
        if not self.aberta:
            return None
        for tipo, c in self.itens:
            if np.linalg.norm(np.array(c) - p) < self.raio:
                return tipo
        return None

    def desenhar(self, img, progresso):
        x, y, l, a = self.linguetaRect()
        if self.aberta:
            fundo = img.copy()
            cv.rectangle(fundo, (0, 0), (self.largura, img.shape[0]), (25, 25, 25), -1)
            cv.addWeighted(fundo, 0.55, img, 0.45, 0, img)
            cv.putText(img, "objetos", (30, 62), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
            for tipo, (cx, cy) in self.itens:
                desenharIcone(img, tipo, (cx, cy), self.raio - 16)
                cv.putText(img, TIPOS[tipo]["nome"], (cx - 34, cy + self.raio - 2), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (230, 230, 230), 1, cv.LINE_AA)
        fundo = img.copy()
        cv.rectangle(fundo, (x, y), (x + l, y + a), (40, 40, 40), -1)
        cv.addWeighted(fundo, 0.7, img, 0.3, 0, img)
        seta = [(x + 17, y + a // 2 - 12), (x + 8, y + a // 2), (x + 17, y + a // 2 + 12)]
        if not self.aberta:
            seta = [(x + 8, y + a // 2 - 12), (x + 17, y + a // 2), (x + 8, y + a // 2 + 12)]
        cv.polylines(img, [np.array(seta)], False, (255, 255, 255), 2, cv.LINE_AA)
        if progresso > 0:
            cv.rectangle(img, (x, y + a + 6), (x + int(l * progresso), y + a + 10), (255, 255, 255), -1)


# ---------------------------------------------------------------- desenho

def sombrear(img, pts, cor):
    pts_i = pts.round().astype(np.int32)
    cv.fillPoly(img, [pts_i], cor, cv.LINE_AA)
    escuro = tuple(int(c * 0.55) for c in cor)
    cv.polylines(img, [pts_i], True, escuro, 3, cv.LINE_AA)


def desenharIcone(img, tipo, centro, maximo=36):
    t = TIPOS[tipo]
    c = np.array(centro, float)
    r = min(t["tamanho"], maximo)
    if t["forma"] == "bola":
        ang = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        pts = c + np.stack([np.cos(ang), np.sin(ang)], 1) * r
    else:
        r = min(r, maximo - 4)
        pts = c + np.array([[-r, -r], [r, -r], [r, r], [-r, r]])
    sombrear(img, pts, t["cor"])
    detalhes(img, tipo, c, r, 0.0)


def detalhes(img, tipo, c, r, angulo):
    """Enfeites que mostram o giro: listra na bola, furos no boliche, barbante no balao, tabuas na caixa."""
    ci = tuple(int(v) for v in c)
    if tipo == "bola":
        d = np.array([math.cos(angulo), math.sin(angulo)]) * r * 0.9
        cv.line(img, tuple((c - d).astype(int)), tuple((c + d).astype(int)), (255, 255, 255), 3, cv.LINE_AA)
    elif tipo == "boliche":
        for k in range(3):
            a = angulo + k * 0.45 - 0.45
            p = c + np.array([math.cos(a - 1.2), math.sin(a - 1.2)]) * r * 0.45
            cv.circle(img, tuple(p.astype(int)), max(3, int(r * 0.1)), (20, 10, 10), -1, cv.LINE_AA)
    elif tipo == "bomba":   # pavio
        topo = c + np.array([r * 0.5, -r * 0.85])
        cv.line(img, tuple(topo.astype(int)), tuple((topo + [r * 0.35, -r * 0.4]).astype(int)), (60, 160, 230), 3, cv.LINE_AA)
    elif tipo == "superbola":
        cv.circle(img, ci, max(3, int(r * 0.55)), (255, 255, 255), 2, cv.LINE_AA)
    elif tipo == "balao":
        base = c + np.array([0, r])
        cv.line(img, tuple(base.astype(int)), tuple((base + [6, r * 1.3]).astype(int)), (220, 220, 220), 1, cv.LINE_AA)
    if tipo != "caixa":   # brilho
        cv.circle(img, (ci[0] - int(r * 0.35), ci[1] - int(r * 0.35)), max(3, int(r * 0.18)), (255, 255, 255), -1, cv.LINE_AA)


def desenharObjeto(img, o, agora):
    pts = o.contorno()
    cor = o.t["cor"]
    if o.pavio is not None:   # bomba armada: pisca cada vez mais rapido
        falta = max(o.pavio - agora, 0.0)
        if math.sin(agora * (8 + 30 * (1 - falta / BOMBA_PAVIO))) > 0:
            cor = (40, 40, 230)
    sombrear(img, pts, cor)
    if o.t["forma"] == "caixa":
        a, b, c_, d = pts[0], pts[1], pts[2], pts[3]
        for k in (1 / 3, 2 / 3):
            p1, p2 = a + (d - a) * k, b + (c_ - b) * k
            cv.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), tuple(int(v * 0.7) for v in o.t["cor"]), 2, cv.LINE_AA)
    detalhes(img, o.tipo, o.pos, o.t["tamanho"], o.body.angle)
    if o.pavio is not None:
        c = tuple(int(v) for v in o.pos)
        texto(img, f"{max(o.pavio - agora, 0):.1f}", (c[0] - 16, c[1] - o.t["tamanho"] - 12), 0.6, (80, 200, 255))


def texto(img, msg, pos, tamanho=0.6, cor=(255, 255, 255)):
    cv.putText(img, msg, pos, cv.FONT_HERSHEY_SIMPLEX, tamanho, (0, 0, 0), 5, cv.LINE_AA)
    cv.putText(img, msg, pos, cv.FONT_HERSHEY_SIMPLEX, tamanho, cor, 2, cv.LINE_AA)


AJUDA = [
    "punho ou pinca na aba: cria  |  no objeto: pega  |  soltar mexendo: joga  |  2 maos: estica / estilingue",
    "mao aberta rapida: tapa  |  rock (indicador+mindinho): ima  |  bomba explode 2.5 s depois de soltar",
    "A aba  G gravidade  S camera lenta  R rastro  C limpa  Z desfaz  D gesto  F tela cheia  H ajuda  Q sai",
]


# ---------------------------------------------------------------- principal

def criarParedes(espaco, w, h):
    """Chao, teto e as duas laterais na borda da imagem. Grossas (120 px) pra nada atravessar rapido."""
    corpo = espaco.static_body
    e = 60   # meia espessura: a face de dentro fica exatamente na borda da tela
    paredes = [pymunk.Segment(corpo, a, b, e) for a, b in
               [((-e, h + e), (w + e, h + e)), ((-e, -e), (-e, h + e)),
                ((w + e, -e), (w + e, h + e)), ((-e, -e), (w + e, -e))]]
    for p in paredes:
        p.elasticity, p.friction = 0.6, 0.8
        p.filter = pymunk.ShapeFilter(categories=CAT_PAREDE)
    espaco.add(*paredes)
    return paredes


def main():
    parser = argparse.ArgumentParser(description="Objetos com fisica controlados pelas maos.")
    parser.add_argument("--fonte", default="webcam", help="webcam, numero da camera ou IP da ESP32")
    parser.add_argument("--largura", type=int, default=1280)
    parser.add_argument("--altura", type=int, default=720)
    parser.add_argument("--cheia", action="store_true")
    parser.add_argument("--sem-espelho", action="store_true")
    args = parser.parse_args()

    baixarModelo()
    mp, detector = criarDetector(num_maos=2)
    try:
        captura, _ = abrirFonte(args.fonte)
    except OSError as e:
        raise SystemExit(f"nao conectou na fonte {args.fonte}: {e}")
    if isinstance(captura, cv.VideoCapture):
        captura.set(cv.CAP_PROP_FRAME_WIDTH, args.largura)
        captura.set(cv.CAP_PROP_FRAME_HEIGHT, args.altura)
        captura.set(cv.CAP_PROP_FPS, 60)   # mais quadros = menos borrao no movimento rapido (se a camera aguentar)

    cv.namedWindow(JANELA, cv.WINDOW_NORMAL)
    cv.resizeWindow(JANELA, args.largura, args.altura)
    cheia = args.cheia
    if cheia:
        cv.setWindowProperty(JANELA, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    espaco = pymunk.Space()
    espaco.gravity = (0, GRAVIDADE)
    espaco.damping = 0.995
    paredes = None
    objetos = []
    maos = {}          # chave -> Mao (gestos)
    fisicas = {}       # chave -> MaoFisica
    aba = None
    ajuda = True
    diagnostico = True   # tecla D: mostra o gesto de cada mao
    modo_g = 0           # indice em MODOS_GRAVIDADE
    lenta = False        # tecla S: camera lenta
    rastro = True        # tecla R
    explosoes = []       # (pos, t0) pra desenhar
    ultimo_ts = -1
    t_anterior = time.time()
    fps = 0.0

    def remover(o):
        if o.restricao is not None:
            espaco.remove(o.restricao)
            o.restricao = None
        espaco.remove(o.body, o.shape)
        objetos.remove(o)

    def tirarRestricao(o):
        if o.restricao is not None:
            espaco.remove(o.restricao)
            o.restricao = None

    def pegar(o, chave, ponto):
        """Prende o objeto na mao pelo ponto onde a pinca pegou."""
        tirarRestricao(o)
        if o.body.body_type != pymunk.Body.DYNAMIC:
            o.body.body_type = pymunk.Body.DYNAMIC
        f = fisicas[chave]
        ancora = o.body.world_to_local(tuple(ponto))
        if o.t["forma"] == "bola" and np.linalg.norm(np.array(ponto) - o.pos) > o.t["tamanho"]:
            ancora = (0, 0)   # pegou pela borda de fora: segura pelo centro
        junta = pymunk.PivotJoint(f.body, o.body, (0, 0), ancora)
        junta.max_force = 150000 * o.t["massa"]   # forte: acompanha a mao mesmo num puxao rapido
        junta.error_bias = pow(1 - 0.5, 60)
        espaco.add(junta)
        o.restricao, o.estado, o.maos = junta, "pego", [chave]
        o.esticar, o.balanco = (1.0, 0.0), None
        f.segurando = o

    def soltarLivre(o, velocidade=None):
        tirarRestricao(o)
        if o.body.body_type != pymunk.Body.DYNAMIC:
            o.body.body_type = pymunk.Body.DYNAMIC
        if velocidade is not None:
            o.body.velocity = tuple(velocidade)
        for chave in o.maos:
            if chave in fisicas and fisicas[chave].segurando is o:
                fisicas[chave].segurando = None
        o.estado, o.maos = "livre", []
        if o.t.get("bomba") and o.pavio is None:
            o.pavio = time.time() + BOMBA_PAVIO

    def explodir(b):
        for chave in list(b.maos):
            if chave in fisicas:
                fisicas[chave].segurando = None
                fisicas[chave].pinca_antes = False
        centro = b.pos
        remover(b)
        explosoes.append((centro, time.time()))
        for o in objetos:
            d = o.pos - centro
            dist = np.linalg.norm(d)
            if dist > BOMBA_RAIO or dist < 1e-3:
                continue
            if o.estado != "livre":
                soltarLivre(o)
            o.body.velocity = tuple(np.array(o.body.velocity) + d / dist * BOMBA_FORCA * (1 - dist / BOMBA_RAIO)
                                    / max(o.t["massa"], 1.0) ** 0.5)
            o.body.angular_velocity += 12 * (1 if d[0] > 0 else -1)

    def comecarEsticar(o, a, b):
        tirarRestricao(o)
        o.body.body_type = pymunk.Body.KINEMATIC
        pa, pb = pontoDe(maos[a], fisicas[a]), pontoDe(maos[b], fisicas[b])
        eixo = pb - pa
        o.gesto = {"dist0": max(np.linalg.norm(eixo), 30.0), "ang0": math.atan2(eixo[1], eixo[0]),
                   "angulo_obj0": o.body.angle, "meios": deque(maxlen=6)}
        o.estado, o.maos = "esticado", [a, b]
        o.balanco = None
        fisicas[a].segurando = fisicas[b].segurando = o

    def objetoEm(p, chave=None):
        info = espaco.point_query_nearest(tuple(p), ALCANCE, SO_OBJETOS)
        if info is not None and info.shape is not None:
            o = next((o for o in objetos if o.shape is info.shape), None)
            if o is not None:
                return o
        # segunda mao: as duas maos nao cabem em cima de um objeto pequeno, entao vale chegar perto
        perto = [(np.linalg.norm(o.pos - p), o) for o in objetos
                 if o.estado == "pego" and o.maos and o.maos[0] != chave]
        perto = [(d, o) for d, o in perto if d < o.t["tamanho"] + ALCANCE_2A_MAO]
        return min(perto, key=lambda x: x[0])[1] if perto else None

    def segura(m, f):
        """Mao fechada OU pinca seguram. Depois de pegar, o ponto segue o gesto usado pra pegar."""
        return m.punho or m.pinca

    def pontoDe(m, f):
        modo = f.modo if f.pinca_antes and f.modo else ("pinca" if m.pinca else "punho")
        return m.ponto if modo == "pinca" else m.centro

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
            if paredes is None:
                paredes = criarParedes(espaco, w, h)
                aba = Aba(h)

            agora = time.time()
            dt_real = min(max(agora - t_anterior, 1 / 120), 1 / 15)
            t_anterior = agora
            dt = dt_real * (0.3 if lenta else 1.0)   # tempo da fisica (camera lenta)

            # ---- maos
            rgb = np.ascontiguousarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            ts = max(int(time.monotonic() * 1000), ultimo_ts + 1)
            ultimo_ts = ts
            resultado = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            for chave in associarMaos(resultado, maos, w, h, agora):
                if chave not in fisicas:
                    fisicas[chave] = MaoFisica(espaco)
            for chave in [k for k, m in maos.items() if agora - m.vista > 1.0]:   # mao que foi embora de vez
                f = fisicas.pop(chave, None)
                if f is not None:
                    if f.segurando is not None and f.segurando in objetos:
                        soltarLivre(f.segurando, f.velocidade())
                    espaco.remove(f.body, f.shape)
                del maos[chave]
            ativas = {k: m for k, m in maos.items() if agora - m.vista <= SUMIU}
            for k in list(fisicas):
                if k in ativas:
                    fisicas[k].mover(pontoDe(ativas[k], fisicas[k]), agora, dt)

            # ---- soltar: so depois da pinca ficar aberta um tempinho, ou da mao sumir de vez.
            # Enquanto a mao some (movimento rapido borra a imagem), o corpo da mao continua
            # na mesma velocidade e o objeto segue a trajetoria: o arremesso nao e cortado.
            soltaram = []
            for k, f in fisicas.items():
                if not f.pinca_antes:
                    continue
                if k in ativas and segura(ativas[k], f):
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
                f = fisicas[k]
                f.pinca_antes, f.solto_em = False, agora
                f.aberta_desde = f.perdida_desde = None
                o = f.segurando
                if o is None or o not in objetos:
                    f.segurando = None
                    continue
                if o.estado == "pego":
                    soltarLivre(o, f.velocidade())
                elif o.estado == "esticado":
                    outra = next(c for c in o.maos if c != k)
                    outra_soltou = outra in soltaram or fisicas[outra].aberta_desde is not None
                    s, eixo = o.esticar
                    meios = o.gesto["meios"]
                    v_meio = ((meios[-1][1] - meios[0][1]) / max(meios[-1][0] - meios[0][0], 1e-3)
                              if len(meios) >= 2 else np.zeros(2))
                    if outra_soltou:
                        soltarLivre(o, v_meio)            # soltou as duas: volta balancando
                        o.balanco = (s, eixo, agora)
                        fisicas[outra].pinca_antes = False
                    else:                                 # soltou uma: vira elastico preso na outra
                        o.body.body_type = pymunk.Body.DYNAMIC
                        o.body.velocity = tuple(v_meio)
                        massa = o.t["massa"]
                        mola = pymunk.DampedSpring(fisicas[outra].body, o.body, (0, 0), (0, 0), 0, 260 * massa, 4 * massa)
                        espaco.add(mola)
                        o.restricao, o.estado, o.maos = mola, "elastico", [outra]
                        fisicas[k].segurando = None
                elif o.estado == "elastico":
                    s, eixo = o.esticar
                    soltarLivre(o)                        # estilingue: sai com a velocidade que tinha
                    o.balanco = (s, eixo, agora)

            # ---- pegar (pinca comecou)
            for k, m in ativas.items():
                f = fisicas[k]
                if not segura(m, f) or f.pinca_antes:
                    continue
                f.pinca_antes = True
                f.modo = "pinca" if m.pinca else "punho"
                p = pontoDe(m, f)
                tipo = aba.itemEm(p)
                if tipo is not None:
                    if len(objetos) >= MAX_OBJETOS:
                        remover(objetos[0])
                    novo = Objeto(espaco, tipo, p)
                    objetos.append(novo)
                    pegar(novo, k, p)
                    continue
                o = objetoEm(p, k)
                if o is None:
                    continue
                if o.estado == "pego" and o.maos and o.maos[0] != k and o.maos[0] in ativas:
                    comecarEsticar(o, o.maos[0], k)       # segunda mao no mesmo objeto: estica
                elif o.estado in ("livre", "elastico", "pego"):
                    soltarLivre(o)
                    pegar(o, k, p)                        # pega (ou pega da outra mao / no ar)

            # ---- esticando: o objeto fica entre as maos
            for o in objetos:
                if o.estado == "esticado":
                    a, b = o.maos
                    if a not in ativas or b not in ativas:
                        continue
                    pa, pb = pontoDe(ativas[a], fisicas[a]), pontoDe(ativas[b], fisicas[b])
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
                    if k in fisicas:
                        d = np.array(fisicas[k].body.position) - o.pos
                        dist = np.linalg.norm(d)
                        o.esticar = (1 + min(dist / (2 * o.t["tamanho"]), 2.0), math.atan2(d[1], d[0]))
                elif o.balanco is not None:
                    s0, eixo, t0 = o.balanco
                    t = agora - t0
                    if t > 1.2:
                        o.esticar, o.balanco = (1.0, 0.0), None
                    else:
                        # balanca em volta do formato normal; o quanto encolhe e limitado pra nao virar do avesso
                        amplitude = min(abs(s0 - 1), 0.5) * (1 if s0 >= 1 else -1)
                        o.esticar = (1 + amplitude * math.exp(-5 * t) * math.cos(22 * t), eixo)

            # ---- tapa: mao aberta e rapida empurra; nao bate no objeto que acabou de jogar
            for k, f in fisicas.items():
                m = ativas.get(k)
                rapida = np.linalg.norm(np.array(f.body.velocity)) > TAPA_VEL
                f.tapa(m is not None and not f.pinca_antes and f.segurando is None and rapida
                       and not m.punho and not m.pinca and agora - f.solto_em > 0.3)

            # ---- ima: sinal do rock puxa os objetos soltos pra mao
            imas = [m.centro for k, m in ativas.items() if m.rock and fisicas[k].segurando is None]
            for c in imas:
                for o in objetos:
                    if o.estado != "livre":
                        continue
                    d = c - o.pos
                    dist = np.linalg.norm(d)
                    if 1 < dist < IMA_RAIO:
                        v = np.array(o.body.velocity) * (0.9 if dist < 120 else 1.0)   # freia perto da mao
                        o.body.velocity = tuple(v + d / dist * IMA_FORCA * dt)

            # ---- bombas
            for o in [o for o in objetos if o.pavio is not None and agora >= o.pavio]:
                explodir(o)

            # ---- fisica
            espaco.gravity = (0, GRAVIDADE * MODOS_GRAVIDADE[modo_g][1])
            for _ in range(SUBPASSOS):
                espaco.step(dt / SUBPASSOS)
            for o in objetos:
                if o.body.body_type != pymunk.Body.DYNAMIC:
                    continue
                v = np.array(o.body.velocity)
                n = np.linalg.norm(v)
                if n > VEL_MAX:   # velocidade maxima (a deteccao da mao as vezes da um pulo)
                    v = v * VEL_MAX / n
                # trava de seguranca: se algo passou da parede, volta pra dentro quicando
                r = o.t["tamanho"]
                x, y = o.body.position
                if x < r or x > w - r:
                    x, v[0] = min(max(x, r), w - r), -v[0] * o.t["quique"]
                if y < r or y > h - r:
                    y, v[1] = min(max(y, r), h - r), -v[1] * o.t["quique"]
                if (x, y) != tuple(o.body.position):
                    o.body.position = (x, y)
                o.body.velocity = tuple(v)
            for o in objetos:
                if np.linalg.norm(np.array(o.body.velocity)) > 350:
                    o.rastro.append(tuple(o.pos.astype(int)))
                elif o.rastro:
                    o.rastro.popleft()

            # ---- abrir/fechar a aba apontando pra lingueta
            apontando = any(m.apontando and aba.naLingueta(m.ponta) for m in ativas.values())
            progresso = 0.0
            if apontando:
                aba.mira_desde = aba.mira_desde or agora
                progresso = min((agora - aba.mira_desde) / ESPERA_ABA, 1.0)
                if progresso >= 1.0:
                    aba.aberta = not aba.aberta
                    aba.mira_desde = agora + 10   # so troca de novo depois de tirar o dedo
            else:
                aba.mira_desde = None

            # ---- tela
            saida = frame.copy()
            if rastro:
                for o in objetos:
                    pts = list(o.rastro)
                    for i in range(1, len(pts)):
                        grossura = max(1, int(o.t["tamanho"] * 0.6 * i / len(pts)))
                        cv.line(saida, pts[i - 1], pts[i], o.t["cor"], grossura, cv.LINE_AA)
            for o in objetos:
                desenharObjeto(saida, o, agora)
            for pos, t0 in list(explosoes):
                t = agora - t0
                if t > 0.4:
                    explosoes.remove((pos, t0))
                    continue
                c = tuple(int(v) for v in pos)
                cv.circle(saida, c, int(40 + BOMBA_RAIO * t / 0.4), (40, 160, 255), max(2, int(14 * (1 - t / 0.4))), cv.LINE_AA)
                cv.circle(saida, c, int(60 * (1 - t / 0.4)) + 1, (200, 240, 255), -1, cv.LINE_AA)
            for c in imas:
                fase = (agora * 2) % 1
                for k in range(3):
                    raio = int(IMA_RAIO * ((fase + k / 3) % 1))
                    cv.circle(saida, tuple(int(v) for v in c), max(raio, 1), (230, 120, 255), 1, cv.LINE_AA)
            aba.desenhar(saida, progresso)
            for k, m in ativas.items():
                f = fisicas[k]
                p = tuple(int(v) for v in pontoDe(m, f))
                cor = (0, 237, 204) if f.segurando is not None else (255, 255, 255)
                if segura(m, f):
                    cv.circle(saida, p, 12 if m.pinca else 22, cor, -1, cv.LINE_AA)
                else:
                    cv.circle(saida, p, 26, cor, 2, cv.LINE_AA)
                if f.shape.filter.mask:   # tapa ligado
                    cv.circle(saida, tuple(int(v) for v in f.body.position), 42, (80, 200, 255), 3, cv.LINE_AA)
                if diagnostico:
                    rotulo = "punho" if m.punho else "PINCA" if m.pinca else "rock (ima)" if m.rock else "aberta"
                    texto(saida, rotulo, (p[0] + 30, p[1] - 14), 0.55,
                          (0, 237, 204) if segura(m, f) else (255, 255, 255))
            for o in objetos:   # elastico: linha da mola
                if o.estado == "elastico" and o.maos and o.maos[0] in fisicas:
                    q = tuple(int(v) for v in fisicas[o.maos[0]].body.position)
                    cv.line(saida, q, tuple(o.pos.astype(int)), (255, 255, 255), 2, cv.LINE_AA)

            gasto = max(time.time() - agora, 1e-3)
            fps = 0.9 * fps + 0.1 / gasto if fps else 1 / gasto
            x0 = aba.largura + 40 if aba.aberta else 40
            texto(saida, f"{fps:.0f} FPS  |  {len(objetos)} objeto(s)  |  gravidade {MODOS_GRAVIDADE[modo_g][0]}"
                         + ("  |  CAMERA LENTA" if lenta else ""),
                  (x0, 34))
            if ajuda:
                for i, linha in enumerate(AJUDA):
                    texto(saida, linha, (x0, h - 20 - (len(AJUDA) - 1 - i) * 26), 0.5)
            cv.imshow(JANELA, saida)

            # ---- teclado
            tecla = cv.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27) or cv.getWindowProperty(JANELA, cv.WND_PROP_VISIBLE) < 1:
                break
            if tecla == ord("a"):
                aba.aberta = not aba.aberta
            elif tecla == ord("g"):
                modo_g = (modo_g + 1) % len(MODOS_GRAVIDADE)
            elif tecla == ord("s"):
                lenta = not lenta
            elif tecla == ord("r"):
                rastro = not rastro
            elif tecla == ord("c"):
                for o in list(objetos):
                    remover(o)
                for f in fisicas.values():
                    f.segurando = None
            elif tecla == ord("z") and objetos:
                remover(objetos[-1])
            elif tecla == ord("h"):
                ajuda = not ajuda
            elif tecla == ord("d"):
                diagnostico = not diagnostico
            elif tecla == ord("f"):
                cheia = not cheia
                cv.setWindowProperty(JANELA, cv.WND_PROP_FULLSCREEN,
                                     cv.WINDOW_FULLSCREEN if cheia else cv.WINDOW_NORMAL)
    finally:
        captura.release()
        detector.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
