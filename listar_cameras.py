"""
Mostra todas as cameras que o PC enxerga, com o numero de cada uma (pra usar em --fonte N).
Inclui cameras virtuais como DroidCam, Iriun e OBS (no Windows elas aparecem pelo DirectShow).

Uso:
    python listar_cameras.py
    (deixe o DroidCam Client CONECTADO no celular antes de rodar)
"""
import os
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")   # sem os avisos de cada numero de camera que nao existe
import cv2 as cv  # noqa: E402
import numpy as np

MAX_CAMERAS = 8


def testar(indice):
    backends = [("DirectShow", cv.CAP_DSHOW), ("MSMF", cv.CAP_MSMF)] if sys.platform.startswith("win") \
        else [("padrao", cv.CAP_ANY)]
    for nome, backend in backends:
        cap = cv.VideoCapture(indice, backend)
        if not cap.isOpened():
            cap.release()
            continue
        frame = None
        for _ in range(10):   # algumas cameras virtuais mandam preto nos primeiros frames
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        cap.release()
        if frame is not None:
            return nome, frame
    return None, None


def main():
    achadas = []
    for i in range(MAX_CAMERAS):
        nome, frame = testar(i)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        escuro = frame.mean() < 8
        print(f"camera {i}: {w}x{h} via {nome}" + ("  (imagem preta: o app esta conectado?)" if escuro else ""))
        miniatura = cv.resize(frame, (400, int(400 * h / w)))
        cv.putText(miniatura, f"--fonte {i}", (12, 36), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv.LINE_AA)
        cv.putText(miniatura, f"--fonte {i}", (12, 36), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
        achadas.append(miniatura)

    if not achadas:
        print("nenhuma camera encontrada")
        return
    altura = max(m.shape[0] for m in achadas)
    linha = np.hstack([cv.copyMakeBorder(m, 0, altura - m.shape[0], 0, 6, cv.BORDER_CONSTANT) for m in achadas])
    cv.imshow("cameras encontradas (qualquer tecla fecha)", linha)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
