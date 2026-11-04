"""
Copia o dataset do YOLO (dataset/) pro formato que o RF-DETR espera, sem mexer no original:

    rf-detr/dataset/
        data.yaml
        train/images  train/labels
        valid/images  valid/labels
        test/images   test/labels    (copia do valid; o RF-DETR tambem avalia no test)

Uso:
    python rf-detr/preparar_dataset.py
"""
import json
import os
import re
import shutil

from comum import DATASET_ORIGINAL, DATASET_RFDETR


def lerClasses():
    """Mesma fonte de verdade do projeto: dataset/classes.json (ou os names do data.yaml)."""
    caminho_json = os.path.join(DATASET_ORIGINAL, "classes.json")
    if os.path.exists(caminho_json):
        with open(caminho_json, encoding="utf-8") as f:
            classes = json.load(f)
    else:
        classes = {}
        with open(os.path.join(DATASET_ORIGINAL, "data.yaml"), encoding="utf-8") as f:
            for linha in f:
                m = re.match(r"^\s+(\d+):\s*(.+?)\s*$", linha)
                if m:
                    classes[m.group(2)] = int(m.group(1))
    if not classes:
        raise SystemExit("nenhuma classe encontrada em dataset/classes.json nem em dataset/data.yaml")
    # lista na ordem do indice (o YOLO usa o indice como id da classe)
    nomes = [None] * (max(classes.values()) + 1)
    for nome, indice in classes.items():
        nomes[indice] = nome
    return [n if n is not None else f"classe_{i}" for i, n in enumerate(nomes)]


def copiarSplit(origem, destino):
    img_o = os.path.join(DATASET_ORIGINAL, "images", origem)
    lbl_o = os.path.join(DATASET_ORIGINAL, "labels", origem)
    img_d = os.path.join(DATASET_RFDETR, destino, "images")
    lbl_d = os.path.join(DATASET_RFDETR, destino, "labels")
    os.makedirs(img_d, exist_ok=True)
    os.makedirs(lbl_d, exist_ok=True)

    total = 0
    if not os.path.isdir(img_o):
        return 0
    for arquivo in sorted(os.listdir(img_o)):
        if not arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(arquivo)[0]
        shutil.copy2(os.path.join(img_o, arquivo), os.path.join(img_d, arquivo))
        label = os.path.join(lbl_o, base + ".txt")
        destino_label = os.path.join(lbl_d, base + ".txt")
        if os.path.exists(label):
            shutil.copy2(label, destino_label)
        else:
            open(destino_label, "w").close()   # imagem de fundo: label vazio
        total += 1
    return total


def separarValidacao(fracao=0.2):
    """Video curto (sem val): passa o fim de cada produto do train pro valid, so na copia."""
    img_t = os.path.join(DATASET_RFDETR, "train", "images")
    por_produto = {}
    for arquivo in sorted(os.listdir(img_t)):
        m = re.match(r"^(.+)_\d+\.(jpg|jpeg|png)$", arquivo, re.IGNORECASE)
        if m:
            por_produto.setdefault(m.group(1), []).append(arquivo)
    movidas = 0
    for arquivos in por_produto.values():
        if len(arquivos) < 2:
            continue
        for arquivo in arquivos[-max(1, round(len(arquivos) * fracao)):]:
            base = os.path.splitext(arquivo)[0]
            shutil.move(os.path.join(img_t, arquivo), os.path.join(DATASET_RFDETR, "valid", "images", arquivo))
            shutil.move(os.path.join(DATASET_RFDETR, "train", "labels", base + ".txt"),
                        os.path.join(DATASET_RFDETR, "valid", "labels", base + ".txt"))
            movidas += 1
    print(f"sem imagens de validacao: separei {movidas} do train (so na copia do RF-DETR)")
    return movidas


def main():
    if os.path.isdir(DATASET_RFDETR):
        shutil.rmtree(DATASET_RFDETR)   # sempre recria do zero a partir do dataset original

    nomes = lerClasses()
    n_train = copiarSplit("train", "train")
    n_val = copiarSplit("val", "valid")
    if n_val == 0 and n_train >= 2:
        n_val = separarValidacao()
        n_train -= n_val
    if n_train == 0 or n_val == 0:
        raise SystemExit(f"dataset insuficiente (train={n_train}, val={n_val}); importe mais videos")
    shutil.copytree(os.path.join(DATASET_RFDETR, "valid"), os.path.join(DATASET_RFDETR, "test"))

    with open(os.path.join(DATASET_RFDETR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {DATASET_RFDETR.replace(os.sep, '/')}\n")
        f.write("train: train/images\nval: valid/images\ntest: test/images\n")
        f.write(f"nc: {len(nomes)}\n")
        f.write("names: [" + ", ".join(f"'{n}'" for n in nomes) + "]\n")

    print(f"dataset pronto em {DATASET_RFDETR}")
    print(f"  classes: {', '.join(nomes)}")
    print(f"  train={n_train}  valid={n_val}  test={n_val} (copia do valid)")


if __name__ == "__main__":
    main()
