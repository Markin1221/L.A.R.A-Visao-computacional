"""
Mostra as deteccoes do RF-DETR numa janela, ao vivo.

Uso:
    python rf-detr/testar.py --fonte webcam                    # pre-treinado no COCO
    python rf-detr/testar.py --fonte 192.168.1.6               # ESP32-CAM
    python rf-detr/testar.py --fonte foto.jpg
    python rf-detr/testar.py --fonte video.mp4 --pesos rf-detr/modelos/melhor.pth
    python rf-detr/testar.py --fonte webcam --segmentar           # pinta o formato do objeto

Teclas: Q ou ESC fecha, S salva o frame em rf-detr/capturas/.
"""
import argparse
import os
import time

import cv2 as cv

from comum import MELHOR_MODELO, PASTA, TAMANHOS, abrirFonte, criarModelo, desenhar, nomesDasDeteccoes, preverBGR


def main():
    parser = argparse.ArgumentParser(description="Testa o RF-DETR em imagem, video, webcam ou ESP32.")
    parser.add_argument("--fonte", default="webcam", help="webcam, numero da camera, arquivo, ou IP/URL da ESP32")
    parser.add_argument("--tamanho", choices=TAMANHOS, default="nano")
    parser.add_argument("--pesos", help=f"checkpoint treinado (ex: {os.path.relpath(MELHOR_MODELO)}). Sem isso usa o COCO")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--otimizar", action="store_true", help="compila o modelo pra inferencia (mais rapido, demora pra iniciar)")
    parser.add_argument("--segmentar", action="store_true", help="usa o modelo de segmentacao: pinta o formato do objeto")
    parser.add_argument("--sem-caixa", action="store_true", help="com --segmentar, mostra so a pintura (sem retangulo)")
    args = parser.parse_args()

    print("carregando o modelo...")
    modelo = criarModelo(args.tamanho, args.pesos, segmentar=args.segmentar)
    caixas = not (args.segmentar and args.sem_caixa)
    titulo = f"RF-DETR {'Seg ' if args.segmentar else ''}{args.tamanho}"
    if args.otimizar and hasattr(modelo, "optimize_for_inference"):
        modelo.optimize_for_inference()
    print("modelo pronto")

    # imagem parada
    if os.path.isfile(args.fonte) and args.fonte.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        frame = cv.imread(args.fonte)
        inicio = time.time()
        det = preverBGR(modelo, frame, args.conf)
        ms = (time.time() - inicio) * 1000
        nomes = nomesDasDeteccoes(det, modelo)
        for nome, conf in sorted(zip(nomes, det.confidence), key=lambda x: -x[1]):
            print(f"  {nome:20s} {conf:.2f}")
        print(f"{len(nomes)} deteccoes em {ms:.0f} ms")
        cv.imshow("RF-DETR", desenhar(frame, det, nomes, caixas))
        cv.waitKey(0)
        return

    try:
        captura, ao_vivo = abrirFonte(args.fonte)
    except OSError as e:
        raise SystemExit(
            f"nao conectou na ESP32 ({e}).\n"
            "Confira: a placa esta ligada? o endereco mudou? (veja no Monitor Serial)\n"
            "o PC esta no mesmo Wi-Fi? outra aba/programa esta usando a camera?"
        )
    pasta_capturas = os.path.join(PASTA, "capturas")
    fps = 0.0
    try:
        while True:
            inicio = time.time()
            ok, frame = captura.read()
            if not ok:
                break
            det = preverBGR(modelo, frame, args.conf)
            saida = desenhar(frame, det, nomesDasDeteccoes(det, modelo), caixas)

            gasto = max(time.time() - inicio, 1e-3)
            fps = 0.9 * fps + 0.1 / gasto if fps else 1 / gasto
            cv.putText(saida, f"{titulo} {fps:.0f} FPS", (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv.putText(saida, f"{titulo} {fps:.0f} FPS", (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.imshow("RF-DETR", saida)

            tecla = cv.waitKey(1 if ao_vivo else 1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("s"):
                os.makedirs(pasta_capturas, exist_ok=True)
                caminho = os.path.join(pasta_capturas, time.strftime("captura_%Y%m%d_%H%M%S.jpg"))
                cv.imwrite(caminho, saida)
                print(f"salvo: {caminho}")
    finally:
        captura.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
