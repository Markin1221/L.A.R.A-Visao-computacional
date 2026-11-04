"""
Compara YOLO x RF-DETR nas mesmas imagens de validacao do dataset.

Mede, pra cada modelo:
  - mAP50: qualidade geral (0 a 1, maior = melhor)
  - precisao / recall / F1 na confianca escolhida
  - tempo medio por imagem e FPS na sua maquina

Uso:
    python rf-detr/comparar.py
    python rf-detr/comparar.py --yolo modelos/melhor.pt --rfdetr rf-detr/modelos/melhor.pth
    python rf-detr/comparar.py --agnostico     # ignora o nome da classe, so avalia se achou o objeto

Use --agnostico quando um dos modelos for pre-treinado no COCO (ele chama de "bottle",
o dataset chama de "garrafa_de_metal", entao por nome nunca bate).
"""
import argparse
import os
import time

import cv2 as cv
import numpy as np

from comum import DATASET_RFDETR, MELHOR_MODELO, PASTA, RAIZ, TAMANHOS, criarModelo, nomesDasDeteccoes, preverBGR

CONF_MINIMA = 0.05   # pega quase tudo pra calcular o mAP


def lerNomes():
    caminho = os.path.join(DATASET_RFDETR, "data.yaml")
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            if linha.startswith("names:"):
                dentro = linha.split("[", 1)[1].rsplit("]", 1)[0]
                return [n.strip().strip("'\"") for n in dentro.split(",")]
    raise SystemExit("rode antes: python rf-detr/preparar_dataset.py")


def lerGT(caminho_label, w, h, nomes):
    caixas = []
    if not os.path.exists(caminho_label):
        return caixas
    with open(caminho_label, encoding="utf-8") as f:
        for linha in f:
            partes = linha.split()
            if len(partes) < 5:
                continue
            c = int(partes[0])
            xc, yc, bw, bh = map(float, partes[1:5])
            caixas.append((nomes[c], [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]))
    return caixas


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0


def casar(preds, gts, chave):
    """preds: [(img, nome, conf, caixa)] ja ordenado por conf. Devolve lista de acertos (1/0) e total de GT."""
    gt_por_img = {img: [b for n, b in lista if chave(n) is not None] for img, lista in gts.items()}
    usados = {img: [False] * len(v) for img, v in gt_por_img.items()}
    acertos = []
    for img, _, _, caixa in preds:
        melhor, j = 0.0, -1
        for k, g in enumerate(gt_por_img.get(img, [])):
            valor = iou(caixa, g)
            if valor > melhor:
                melhor, j = valor, k
        if melhor >= 0.5 and not usados[img][j]:
            usados[img][j] = True
            acertos.append(1)
        else:
            acertos.append(0)
    return acertos, sum(len(v) for v in gt_por_img.values())


def avaliar(preds, gts, conf_corte, agnostico):
    classes = {"objeto"} if agnostico else {n for lista in gts.values() for n, _ in lista}
    aps, tp_t, fp_t, pos_t = [], 0, 0, 0
    for classe in classes:
        mesma = (lambda n: n) if agnostico else (lambda n, c=classe: n if n == c else None)
        p = sorted([x for x in preds if mesma(x[1]) is not None], key=lambda x: -x[2])
        g = {img: [(n, b) for n, b in lista if mesma(n) is not None] for img, lista in gts.items()}

        acertos, positivos = casar(p, g, lambda n: n)
        if positivos == 0:
            continue
        tp = np.cumsum(acertos) if acertos else np.array([0])
        fp = np.cumsum([1 - a for a in acertos]) if acertos else np.array([0])
        recall = tp / positivos
        precisao = tp / np.maximum(tp + fp, 1e-9)
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[0.0], precisao, [0.0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        aps.append(float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])))

        acima = [a for a, x in zip(acertos, p) if x[2] >= conf_corte]
        tp_t += sum(acima)
        fp_t += len(acima) - sum(acima)
        pos_t += positivos

    prec = tp_t / (tp_t + fp_t) if tp_t + fp_t else 0.0
    rec = tp_t / pos_t if pos_t else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return (float(np.mean(aps)) if aps else 0.0), prec, rec, f1


def rodar(nome, prever, imagens, gts):
    """Roda o modelo em todas as imagens, medindo o tempo (as 3 primeiras rodadas sao aquecimento)."""
    for _ in range(3):
        prever(imagens[0][1])
    preds, tempos = [], []
    for caminho, frame in imagens:
        inicio = time.perf_counter()
        resultado = prever(frame)
        tempos.append(time.perf_counter() - inicio)
        for nome_cls, conf, caixa in resultado:
            preds.append((caminho, nome_cls, conf, caixa))
    ms = 1000 * float(np.mean(tempos))
    print(f"  {nome}: {len(preds)} deteccoes, {ms:.1f} ms/imagem")
    return preds, ms


def main():
    parser = argparse.ArgumentParser(description="Compara YOLO e RF-DETR no dataset de validacao.")
    parser.add_argument("--yolo", default=os.path.join(RAIZ, "modelos", "melhor.pt"))
    parser.add_argument("--rfdetr", default=MELHOR_MODELO, help="checkpoint .pth (use 'coco' pro pre-treinado)")
    parser.add_argument("--tamanho", choices=TAMANHOS, default="nano")
    parser.add_argument("--conf", type=float, default=0.5, help="corte de confianca pra precisao/recall")
    parser.add_argument("--agnostico", action="store_true", help="ignora o nome da classe")
    args = parser.parse_args()

    nomes = lerNomes()
    pasta_img = os.path.join(DATASET_RFDETR, "valid", "images")
    pasta_lbl = os.path.join(DATASET_RFDETR, "valid", "labels")
    imagens, gts = [], {}
    for arquivo in sorted(os.listdir(pasta_img)):
        frame = cv.imread(os.path.join(pasta_img, arquivo))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        imagens.append((arquivo, frame))
        gts[arquivo] = lerGT(os.path.join(pasta_lbl, os.path.splitext(arquivo)[0] + ".txt"), w, h, nomes)
    n_obj = sum(len(v) for v in gts.values())
    print(f"{len(imagens)} imagens de validacao, {n_obj} objetos marcados")
    if len(imagens) < 20:
        print("AVISO: com tao poucas imagens o resultado varia muito. Grave mais videos pra uma comparacao justa.")

    resultados = []

    # YOLO
    if os.path.exists(args.yolo):
        from ultralytics import YOLO
        yolo = YOLO(args.yolo)

        def prever_yolo(frame):
            r = yolo(frame, conf=CONF_MINIMA, verbose=False)[0]
            return [(r.names[int(c)], float(p), b) for b, c, p in
                    zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]

        print(f"YOLO: {os.path.relpath(args.yolo, RAIZ)}")
        preds, ms = rodar("YOLO", prever_yolo, imagens, gts)
        resultados.append(("YOLO", preds, ms))
    else:
        print(f"YOLO pulado: {args.yolo} nao existe")

    # RF-DETR
    pesos = None if args.rfdetr == "coco" else args.rfdetr
    if pesos and not os.path.exists(pesos):
        print(f"RF-DETR pulado: {pesos} nao existe (treine com rf-detr/treinar.py ou use --rfdetr coco)")
    else:
        rf = criarModelo(args.tamanho, pesos)

        def prever_rf(frame):
            det = preverBGR(rf, frame, CONF_MINIMA)
            return [(n, float(p), b) for n, p, b in
                    zip(nomesDasDeteccoes(det, rf), det.confidence.tolist(), det.xyxy.tolist())]

        print(f"RF-DETR {args.tamanho}: {'pre-treinado COCO' if not pesos else os.path.relpath(pesos, RAIZ)}")
        preds, ms = rodar("RF-DETR", prever_rf, imagens, gts)
        resultados.append((f"RF-DETR {args.tamanho}", preds, ms))

    if not resultados:
        raise SystemExit("nenhum modelo pra comparar")

    nomes_previstos = {p[1] for _, preds, _ in resultados for p in preds}
    if not args.agnostico and not (nomes_previstos & set(nomes)):
        print("AVISO: nenhum nome de classe bate com o dataset. Se usar modelo COCO, rode com --agnostico.")

    print()
    print(f"{'modelo':<18}{'mAP50':>8}{'precisao':>10}{'recall':>8}{'F1':>7}{'ms/img':>9}{'FPS':>7}")
    print("-" * 67)
    linhas = []
    for nome, preds, ms in resultados:
        ap, prec, rec, f1 = avaliar(preds, gts, args.conf, args.agnostico)
        linha = f"{nome:<18}{ap:>8.3f}{prec:>10.3f}{rec:>8.3f}{f1:>7.3f}{ms:>9.1f}{1000 / ms:>7.1f}"
        print(linha)
        linhas.append(linha)
    print(f"\n(precisao/recall/F1 com confianca >= {args.conf}; mAP50 com IoU 0.5"
          f"{'; sem considerar a classe' if args.agnostico else ''})")

    caminho = os.path.join(PASTA, "resultado_comparacao.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(time.strftime("%d/%m/%Y %H:%M") + f" | {len(imagens)} imagens, {n_obj} objetos\n")
        f.write("\n".join(linhas) + "\n")
    print(f"salvo em {os.path.relpath(caminho, RAIZ)}")


if __name__ == "__main__":
    main()
