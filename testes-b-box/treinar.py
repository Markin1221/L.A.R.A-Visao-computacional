"""
Treina (ou re-treina) o YOLO com o dataset atual.

Uso:
    python testes-b-box/treinar.py
    python testes-b-box/treinar.py --epocas 100 --base testes-yolo/yolo11n.pt

Se ja existir modelos/melhor.pt, ele e usado como ponto de partida (fine-tune),
o que deixa o re-treino bem mais rapido. O resultado final vai pra modelos/melhor.pt.
"""
import argparse
import os
import re
import shutil
from collections import defaultdict

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(RAIZ, "dataset", "data.yaml")
PASTA_MODELOS = os.path.join(RAIZ, "modelos")
MELHOR_MODELO = os.path.join(PASTA_MODELOS, "melhor.pt")
MODELO_INICIAL = os.path.join(RAIZ, "testes-yolo", "yolo11n.pt")
PASTA_RUNS = os.path.join(RAIZ, "runs", "treino")


def contarImagens(split):
    pasta = os.path.join(RAIZ, "dataset", "images", split)
    if not os.path.isdir(pasta):
        return 0
    return sum(1 for f in os.listdir(pasta) if f.lower().endswith(".jpg"))


def garantirVal(fracao=0.2):
    """
    Se o val estiver vazio (videos curtos), move as ultimas ~20% imagens de cada
    produto do train pro val, junto com os labels. Pega um trecho seguido do fim
    pra evitar frames quase iguais nos dois lados.
    """
    img_train = os.path.join(RAIZ, "dataset", "images", "train")
    lbl_train = os.path.join(RAIZ, "dataset", "labels", "train")
    img_val = os.path.join(RAIZ, "dataset", "images", "val")
    lbl_val = os.path.join(RAIZ, "dataset", "labels", "val")
    os.makedirs(img_val, exist_ok=True)
    os.makedirs(lbl_val, exist_ok=True)

    por_produto = defaultdict(list)
    for arquivo in sorted(os.listdir(img_train)):
        m = re.match(r"^(.+)_\d+\.jpg$", arquivo)
        if m:
            por_produto[m.group(1)].append(arquivo)

    movidas = 0
    for produto, arquivos in por_produto.items():
        if len(arquivos) < 2:
            continue  # precisa sobrar pelo menos 1 no train
        qtd = max(1, round(len(arquivos) * fracao))
        for arquivo in arquivos[-qtd:]:
            base = os.path.splitext(arquivo)[0]
            shutil.move(os.path.join(img_train, arquivo), os.path.join(img_val, arquivo))
            label = os.path.join(lbl_train, base + ".txt")
            if os.path.exists(label):
                shutil.move(label, os.path.join(lbl_val, base + ".txt"))
            movidas += 1
    print(f"val estava vazio: {movidas} imagens movidas do train pro val")


def treinar(epocas=50, imgsz=640, base=None):
    # import aqui dentro: ultralytics/torch demoram pra carregar
    import torch
    from ultralytics import YOLO

    n_train, n_val = contarImagens("train"), contarImagens("val")
    if n_train >= 2 and n_val == 0:
        garantirVal()
        n_train, n_val = contarImagens("train"), contarImagens("val")
    if n_train == 0 or n_val == 0:
        print(f"dataset insuficiente (train={n_train}, val={n_val}); importe mais videos antes de treinar")
        return None

    if base is None:
        base = MELHOR_MODELO if os.path.exists(MELHOR_MODELO) else MODELO_INICIAL

    gpu = torch.cuda.is_available()
    print(f"treinando a partir de {base} | train={n_train} val={n_val} | {'GPU' if gpu else 'CPU'}")

    modelo = YOLO(base)
    modelo.train(
        data=DATA_YAML,
        epochs=epocas,
        imgsz=imgsz,
        batch=-1 if gpu else 8,     # -1 = ajusta o batch pela memoria da GPU
        device=0 if gpu else "cpu",
        patience=15,                # para cedo se nao melhorar
        project=PASTA_RUNS,
        name="treino",              # vira treino, treino2, treino3...
        exist_ok=False,
    )

    best = os.path.join(str(modelo.trainer.save_dir), "weights", "best.pt")
    if not os.path.exists(best):
        print("treino terminou sem gerar best.pt")
        return None

    os.makedirs(PASTA_MODELOS, exist_ok=True)
    shutil.copyfile(best, MELHOR_MODELO)
    print(f"modelo salvo em {MELHOR_MODELO}")
    return MELHOR_MODELO


def main():
    parser = argparse.ArgumentParser(description="Treina o YOLO com o dataset.")
    parser.add_argument("--epocas", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base", help="modelo de partida (padrao: modelos/melhor.pt ou yolo11n.pt)")
    args = parser.parse_args()
    treinar(args.epocas, args.imgsz, args.base)


if __name__ == "__main__":
    main()
