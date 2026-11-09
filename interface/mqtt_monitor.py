"""
Mostra no terminal tudo o que o L.A.R.A publica no MQTT (pra testar sem app no celular).

    python interface/mqtt_monitor.py
    python interface/mqtt_monitor.py --so-contagem
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mqtt_lara import HOST, PORTA, PREFIXO  # noqa: E402

import paho.mqtt.client as mqtt  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--so-contagem", action="store_true", help="mostra so o estoque")
    args = parser.parse_args()
    filtro = f"{PREFIXO}/contagem" if args.so_contagem else f"{PREFIXO}/#"

    def conectou(cliente, *_):
        print(f"ouvindo {filtro} em {HOST}:{PORTA}  (Ctrl+C sai)\n")
        cliente.subscribe(filtro)

    def mensagem(_c, _u, msg):
        texto = msg.payload.decode(errors="replace")
        try:
            dados = json.loads(texto)
            if msg.topic.endswith("/contagem") and isinstance(dados, dict):
                itens = "  ".join(f"{p}: {n}" for p, n in dados.get("produtos", {}).items()) or "(vazio)"
                texto = f"[{dados.get('hora', '')}] total {dados.get('total', 0)}  |  {itens}"
            elif isinstance(dados, (dict, list)):
                texto = json.dumps(dados, ensure_ascii=False)
        except ValueError:
            pass
        print(f"{msg.topic:<32} {texto}")

    try:
        cliente = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        cliente = mqtt.Client()
    cliente.on_connect = conectou
    cliente.on_message = mensagem
    cliente.connect(HOST, PORTA, 30)
    try:
        cliente.loop_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
