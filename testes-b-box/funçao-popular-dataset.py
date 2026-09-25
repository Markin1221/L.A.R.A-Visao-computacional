import cv2 as cv
import os


def popularDataset(caminho_arquivo, nome_produto):
    nome_arquivo = nome_produto.strip().lower().replace(" ", "_")

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

    numero_frame_real = -1 
    numero_imagem = 0       

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
                f"0 {x_centro:.6f} {y_centro:.6f} {largura_normalizada:.6f} {altura_normalizada:.6f}"
            )

        cv.imwrite(nome_imagem, frame_o)
        numero_imagem += 1

    video.release()
    print(f"{numero_imagem} imagens salvas")


if __name__ == "__main__":
    popularDataset("testes-b-box\\video2.mp4", "garrafa")