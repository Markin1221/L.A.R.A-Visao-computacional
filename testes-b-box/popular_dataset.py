"""
Gera um dataset no formato YOLO a partir de um video de UM produto sobre fundo liso.

Uso (rodando de qualquer pasta):
    python testes-b-box/popular_dataset.py caminho/video.mp4 "coca cola 2l"
    python testes-b-box/popular_dataset.py caminho/video.mp4 "doritos" --preview
    python testes-b-box/popular_dataset.py caminho/video_sem_produto.mp4 --fundo
"""
import argparse
import json
import os
import re

import cv2 as cv

# caminhos sempre a partir da raiz do repo, independente de onde o script e rodado
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(RAIZ, "dataset")
CLASSES_JSON = os.path.join(DATASET_DIR, "classes.json")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
PREVIEW_DIR = os.path.join(DATASET_DIR, "preview")

AREA_MIN = 0.01          # bbox menor que 1% da imagem = ruido
AREA_MAX = 0.90          # bbox maior que 90% da imagem = pegou o fundo/borda
IMAGENS_POR_BLOCO = 7    # imagens seguidas ficam no mesmo split (evita vazamento)
BLOCOS_POR_CICLO = 5     # 1 bloco a cada 5 vai pra val (~20%)


def normalizarNome(nome):
    """'Coca Cola 2L ' -> 'coca_cola_2l' (mesmo formato pra classe e pra arquivo)."""
    return "_".join(nome.strip().lower().split())


def carregarClasses():
    """Le classes.json; se nao existir, reaproveita os names do data.yaml."""
    if os.path.exists(CLASSES_JSON):
        with open(CLASSES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

    classes = {}
    if os.path.exists(DATA_YAML):
        with open(DATA_YAML, "r", encoding="utf-8") as f:
            for linha in f:
                m = re.match(r"^\s+(\d+):\s*(.+?)\s*$", linha)
                if m:
                    classes[normalizarNome(m.group(2))] = int(m.group(1))
    return classes


def escreverDataYaml(classes):
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


def registrarClasse(nome_produto):
    """Garante um indice pro produto, salva classes.json e data.yaml, devolve o indice."""
    nome = normalizarNome(nome_produto)
    classes = carregarClasses()

    if nome not in classes:
        classes[nome] = max(classes.values(), default=-1) + 1

    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(CLASSES_JSON, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    escreverDataYaml(classes)

    return classes[nome]


def detectarBbox(frame):
    """Devolve (x, y, w, h) em pixels do maior objeto, ou None se o resultado nao parecer valido."""
    cinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cinza = cv.GaussianBlur(cinza, (5, 5), 0)

    _, binario = cv.threshold(cinza, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binario = cv.morphologyEx(binario, cv.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv.findContours(binario, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    x, y, w, h = cv.boundingRect(max(contornos, key=cv.contourArea))
    altura_img, largura_img = cinza.shape
    proporcao = (w * h) / (largura_img * altura_img)

    if not (AREA_MIN <= proporcao <= AREA_MAX):
        return None
    return x, y, w, h


def proximoIndice(prefixo):
    """Continua a numeracao de onde parou, pra nao sobrescrever imagens de videos anteriores."""
    maior = -1
    padrao = re.compile(rf"^{re.escape(prefixo)}_(\d+)\.jpg$")
    for split in ("train", "val"):
        pasta = os.path.join(DATASET_DIR, "images", split)
        if not os.path.isdir(pasta):
            continue
        for arquivo in os.listdir(pasta):
            m = padrao.match(arquivo)
            if m:
                maior = max(maior, int(m.group(1)))
    return maior + 1


def popularDataset(caminho_video, nome_produto=None, fundo=False, intervalo_seg=0.5, preview=False):
    if fundo:
        prefixo = "fundo"
        indice_classe = None
    else:
        prefixo = normalizarNome(nome_produto)
        indice_classe = registrarClasse(nome_produto)

    pastas = {}
    for split in ("train", "val"):
        pastas[split] = (
            os.path.join(DATASET_DIR, "images", split),
            os.path.join(DATASET_DIR, "labels", split),
        )
        for pasta in pastas[split]:
            os.makedirs(pasta, exist_ok=True)
    if preview:
        os.makedirs(PREVIEW_DIR, exist_ok=True)

    video = cv.VideoCapture(caminho_video)
    if not video.isOpened():
        raise RuntimeError(f"nao deu pra abrir o video: {caminho_video}")

    fps = video.get(cv.CAP_PROP_FPS) or 30
    passo = max(1, round(fps * intervalo_seg))   # amostra por tempo, nao por n de frames

    numero_frame = -1
    numero_imagem = proximoIndice(prefixo)
    salvas = descartadas = 0

    while True:
        # grab() avanca sem decodificar; so decodifica (retrieve) os frames usados
        if not video.grab():
            break
        numero_frame += 1
        if numero_frame % passo != 0:
            continue

        ok, frame = video.retrieve()
        if not ok:
            break

        if fundo:
            linha_label = ""   # imagem negativa: label vazio
            bbox = None
        else:
            bbox = detectarBbox(frame)
            if bbox is None:
                descartadas += 1
                continue
            x, y, w, h = bbox
            altura_img, largura_img = frame.shape[:2]
            linha_label = (
                f"{indice_classe} {(x + w / 2) / largura_img:.6f} {(y + h / 2) / altura_img:.6f} "
                f"{w / largura_img:.6f} {h / altura_img:.6f}\n"
            )

        bloco = numero_imagem // IMAGENS_POR_BLOCO
        split = "val" if bloco % BLOCOS_POR_CICLO == BLOCOS_POR_CICLO - 1 else "train"
        pasta_img, pasta_txt = pastas[split]
        base = f"{prefixo}_{numero_imagem:06d}"

        # imagem primeiro: se falhar, nao fica label orfao
        if not cv.imwrite(os.path.join(pasta_img, base + ".jpg"), frame):
            print(f"falha ao salvar {base}.jpg, pulando")
            continue
        with open(os.path.join(pasta_txt, base + ".txt"), "w", encoding="utf-8") as f:
            f.write(linha_label)

        if preview and bbox is not None:
            x, y, w, h = bbox
            img_prev = frame.copy()
            cv.rectangle(img_prev, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.imwrite(os.path.join(PREVIEW_DIR, base + ".jpg"), img_prev)

        numero_imagem += 1
        salvas += 1

    video.release()
    print(f"{salvas} imagens salvas, {descartadas} frames descartados (bbox invalido)")


def main():
    parser = argparse.ArgumentParser(description="Gera dataset YOLO a partir de um video.")
    parser.add_argument("video", help="caminho do video")
    parser.add_argument("produto", nargs="?", help='nome do produto, ex: "coca cola 2l"')
    parser.add_argument("--fundo", action="store_true", help="video SEM produto (gera imagens negativas)")
    parser.add_argument("--intervalo", type=float, default=0.5, help="segundos entre frames salvos (padrao 0.5)")
    parser.add_argument("--preview", action="store_true", help="salva copias com o bbox desenhado em dataset/preview")
    args = parser.parse_args()

    if not args.fundo and not args.produto:
        parser.error("informe o nome do produto (ou use --fundo)")

    popularDataset(args.video, args.produto, args.fundo, args.intervalo, args.preview)


if __name__ == "__main__":
    main()
