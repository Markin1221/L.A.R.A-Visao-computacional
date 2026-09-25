import cv2 as cv
import json
import os

DATASET_DIR = r"C:\Users\marki\OneDrive\Documentos\GitHub\L.A.R.A-Visao-computacional\dataset"
CLASSES_JSON = os.path.join(DATASET_DIR, "classes.json")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")


def registrarClasse(nome_produto):
    """
    Le dataset/classes.json, garante que nome_produto tenha um indice
    (cria um novo se for produto inedito), salva o json de volta,
    reescreve o data.yaml e devolve o indice da classe.
    """
    nome_produto = nome_produto.strip().lower()

    if os.path.exists(CLASSES_JSON):
        with open(CLASSES_JSON, "r", encoding="utf-8") as f:
            classes = json.load(f)
    else:
        classes = {}

    if nome_produto not in classes:
        classes[nome_produto] = len(classes)  # proximo indice livre

    with open(CLASSES_JSON, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    escreverDataYaml(classes)

    return classes[nome_produto]


def escreverDataYaml(classes):
    # ordena pelo indice, pra o names sair 0, 1, 2... na ordem certa
    nomes_ordenados = sorted(classes.items(), key=lambda item: item[1])

    linhas = [
        f"path: {DATASET_DIR.replace(os.sep, '/')}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for nome, indice in nomes_ordenados:
        linhas.append(f"  {indice}: {nome}")

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


def popularDataset(caminho_arquivo, nome_produto):
    nome_arquivo = nome_produto.strip().lower().replace(" ", "_")
    indice_classe = registrarClasse(nome_produto)

    val_img_dir = "dataset\\images\\val"
    train_img_dir = "dataset\\images\\train"
    val_txt_dir = "dataset\\labels\\val"
    train_txt_dir = "dataset\\labels\\train"

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_txt_dir, exist_ok=True)
    os.makedirs(train_txt_dir, exist_ok=True)

    video = cv.VideoCapture(caminho_arquivo)

    if not video.isOpened():
        raise RuntimeError("não deu pra abrir o video")

    numero_frame_real = -1  # sobe 1 a cada frame lido, então o primeiro vira 0
    numero_imagem = 0       # só sobe quando uma imagem é salva

    while True:
        sucesso, frame_o = video.read()

        if not sucesso:
            break

        numero_frame_real += 1

        if numero_frame_real % 5 != 0:
            continue

        frame = cv.cvtColor(frame_o, cv.COLOR_BGR2GRAY)

        ret, binario = cv.threshold(
            frame, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

        contornos, hierarquia = cv.findContours(
            binario, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        if not contornos:
            continue

        contorno = max(contornos, key=cv.contourArea)

        x, y, largura, altura = cv.boundingRect(contorno)
        altura_da_imagem, largura_da_imagem = frame.shape

        x_centro = (x + largura / 2) / largura_da_imagem
        y_centro = (y + altura / 2) / altura_da_imagem
        largura_normalizada = largura / largura_da_imagem
        altura_normalizada = altura / altura_da_imagem

        bloco = numero_imagem // 7
        split = "val" if bloco % 5 == 4 else "train"

        if split == "val":
            output_img_dir = val_img_dir
            output_txt_dir = val_txt_dir
        else:
            output_img_dir = train_img_dir
            output_txt_dir = train_txt_dir

        nome_imagem = os.path.join(output_img_dir, f"{nome_arquivo}_{numero_imagem:06d}.jpg")
        nome_txt = os.path.join(output_txt_dir, f"{nome_arquivo}_{numero_imagem:06d}.txt")

        with open(nome_txt, "w", encoding="utf-8") as arquivo:
            arquivo.write(
                f"{indice_classe} {x_centro:.6f} {y_centro:.6f} {largura_normalizada:.6f} {altura_normalizada:.6f}"
            )

        cv.imwrite(nome_imagem, frame_o)
        numero_imagem += 1

    video.release()
    print(f"{numero_imagem} imagens salvas")