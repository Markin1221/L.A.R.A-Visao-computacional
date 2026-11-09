"""
MQTT do L.A.R.A: publica o que a camera reconhece e a contagem de cada produto (estoque).

Topicos (PREFIXO = "lara" por padrao):
  lara/status               "online" / "offline"  (retido; o "offline" o broker manda sozinho se o L.A.R.A cair)
  lara/deteccoes            JSON com as deteccoes do momento (no maximo 1 por segundo no ao vivo)
  lara/contagem             JSON {"produto": quantidade, ...}  (retido: quem conectar depois ja recebe)
  lara/contagem/<produto>   so o numero daquele produto            (retido)

A contagem do ao vivo e estabilizada: usa o valor que mais apareceu no ultimo 1,5 s,
pra um frame ruim nao fazer o estoque "piscar". So publica quando muda (ou a cada 15 s).

Configuracao (variaveis de ambiente, todas opcionais):
  LARA_MQTT_HOST      broker (padrao: broker.hivemq.com, publico e gratis; pra rede local: localhost)
  LARA_MQTT_PORTA     1883
  LARA_MQTT_PREFIXO   lara
  LARA_MQTT_USUARIO / LARA_MQTT_SENHA   se o broker pedir
  LARA_MQTT=0         desliga o MQTT

Precisa de: pip install paho-mqtt
"""
import json
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime

HOST = os.environ.get("LARA_MQTT_HOST", "broker.hivemq.com")
PORTA = int(os.environ.get("LARA_MQTT_PORTA", "1883"))
PREFIXO = os.environ.get("LARA_MQTT_PREFIXO", "lara").strip("/")
JANELA_S = 1.5          # s de historico pra estabilizar a contagem
INTERVALO_DETECCOES = 1.0
REPETE_CONTAGEM = 15.0  # republica mesmo sem mudar (mostra que ta vivo)


def _agora():
    return datetime.now().isoformat(timespec="seconds")


def contar(deteccoes):
    """[{'classe': 'doritos', ...}, ...] -> {'doritos': 2, ...}
    Com profundidade, cada deteccao (a frente de uma fileira) vale 'quantidade' produtos."""
    total = Counter()
    for d in deteccoes:
        total[d["classe"]] += int(d.get("quantidade", 1))
    return dict(total)


class Estabilizador:
    """Guarda as contagens recentes e devolve, por produto, o valor mais comum na janela."""

    def __init__(self, janela=JANELA_S):
        self.janela = janela
        self.historico = deque()

    def adicionar(self, contagem, agora=None):
        agora = time.time() if agora is None else agora
        self.historico.append((agora, contagem))
        while self.historico and agora - self.historico[0][0] > self.janela:
            self.historico.popleft()
        classes = set().union(*(c for _, c in self.historico))
        estavel = {}
        for classe in sorted(classes):
            valores = Counter(c.get(classe, 0) for _, c in self.historico)
            n = max(valores.items(), key=lambda kv: (kv[1], kv[0]))[0]   # empate: o maior
            if n > 0:
                estavel[classe] = n
        return estavel


class LaraMQTT:
    def __init__(self):
        self.conectado = False
        self.erro = None
        self.cliente = None
        self.trava = threading.Lock()
        self.ultima_contagem = {}         # estavel, o que foi publicado por ultimo
        self.produtos_vistos = set()      # pra publicar 0 quando um produto some
        self.publicada_em = 0.0
        self.deteccoes_em = 0.0
        self.estabilizador = Estabilizador()
        if os.environ.get("LARA_MQTT", "1") == "0":
            self.erro = "desligado (LARA_MQTT=0)"
            return
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self.erro = "instale com: pip install paho-mqtt"
            print("MQTT desligado:", self.erro)
            return
        try:   # paho-mqtt 2.x
            self.cliente = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                       client_id=f"lara-{os.getpid()}-{int(time.time()) % 10000}")
        except AttributeError:   # paho-mqtt 1.x
            self.cliente = mqtt.Client(client_id=f"lara-{os.getpid()}")
        usuario = os.environ.get("LARA_MQTT_USUARIO")
        if usuario:
            self.cliente.username_pw_set(usuario, os.environ.get("LARA_MQTT_SENHA"))
        self.cliente.will_set(self.topico("status"), "offline", qos=1, retain=True)
        self.cliente.on_connect = self._conectou
        self.cliente.on_disconnect = self._caiu
        self.cliente.reconnect_delay_set(1, 30)
        try:
            self.cliente.connect_async(HOST, PORTA, keepalive=30)
            self.cliente.loop_start()   # thread propria: reconecta sozinho
        except Exception as e:
            self.erro = str(e)
            print("MQTT: nao conectou em", HOST, "-", e)

    # ------------------------------------------------------------ conexao
    def topico(self, *partes):
        return "/".join([PREFIXO, *partes])

    def _conectou(self, cliente, _ud, _flags, codigo, *_):
        ok = (codigo == 0) if isinstance(codigo, int) else not codigo.is_failure
        self.conectado, self.erro = ok, None if ok else f"broker recusou ({codigo})"
        if ok:
            print(f"MQTT conectado em {HOST}:{PORTA}  (topicos {PREFIXO}/...)")
            cliente.publish(self.topico("status"), "online", qos=1, retain=True)
            if self.ultima_contagem:
                self._publicarContagem(self.ultima_contagem)

    def _caiu(self, *_):
        self.conectado = False

    def info(self):
        return {"ativo": self.cliente is not None, "conectado": self.conectado, "erro": self.erro,
                "broker": f"{HOST}:{PORTA}", "prefixo": PREFIXO, "contagem": self.ultima_contagem}

    def _publicar(self, topico, dados, retain=False):
        if self.cliente is None:
            return
        texto = dados if isinstance(dados, str) else json.dumps(dados, ensure_ascii=False)
        self.cliente.publish(topico, texto, qos=0, retain=retain)   # se estiver offline, paho guarda na fila

    # ------------------------------------------------------------ publicacao
    def _publicarContagem(self, contagem):
        completa = {p: contagem.get(p, 0) for p in sorted(self.produtos_vistos | set(contagem))}
        self._publicar(self.topico("contagem"), {"hora": _agora(), "total": sum(completa.values()),
                                                 "produtos": completa}, retain=True)
        for produto, n in completa.items():
            self._publicar(self.topico("contagem", produto), str(n), retain=True)

    def publicarDeteccoes(self, deteccoes, fonte, ao_vivo=True):
        """Chamado a cada resultado do modelo. deteccoes: [{'classe','conf',...}]."""
        agora = time.time()
        with self.trava:
            if not ao_vivo or agora - self.deteccoes_em >= INTERVALO_DETECCOES:
                self.deteccoes_em = agora
                self._publicar(self.topico("deteccoes"), {
                    "hora": _agora(), "fonte": fonte,
                    "deteccoes": [{"classe": d["classe"], "conf": round(float(d["conf"]), 3)} for d in deteccoes]})

            contagem = contar(deteccoes)
            estavel = self.estabilizador.adicionar(contagem, agora) if ao_vivo else contagem
            self.produtos_vistos |= set(estavel)
            if estavel != self.ultima_contagem or agora - self.publicada_em >= REPETE_CONTAGEM:
                self.ultima_contagem = estavel
                self.publicada_em = agora
                self._publicarContagem(estavel)
            return estavel

    def reiniciarAoVivo(self):
        """Novo stream: comeca a janela do zero (nao mistura com a camera anterior)."""
        with self.trava:
            self.estabilizador = Estabilizador()

    def fechar(self):
        if self.cliente is not None:
            self._publicar(self.topico("status"), "offline", retain=True)
            self.cliente.loop_stop()
            self.cliente.disconnect()
