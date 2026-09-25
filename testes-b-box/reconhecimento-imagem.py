import cv2 as cv
import numpy as np
import os

img_caminho = 'testes-b-box\\fotos\\foto-garrafa001.jpeg'

img = cv.imread(
    img_caminho,
    cv.IMREAD_GRAYSCALE
)

kernel = np.ones((5,5), np.uint8)

erosion = cv.erode(img, kernel, iterations=1)

img2 = cv.imread(
    'testes-b-box\\fotos\\foto-garrafa002.jpeg',
    cv.IMREAD_GRAYSCALE
)

erosion2 = cv.erode(img2, kernel, iterations=1)

diff = cv.absdiff(img, img2)

ret, binario = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

#cv.namedWindow("Original", cv.WINDOW_NORMAL)
#cv.resizeWindow("Original", 600, 400)

#cv.namedWindow("Erosion", cv.WINDOW_NORMAL)
#cv.resizeWindow("Erosion", 600, 400)

#cv.namedWindow("Diferenca", cv.WINDOW_NORMAL)
#cv.resizeWindow("Diferenca", 600, 400)

#cv.namedWindow("binario", cv.WINDOW_NORMAL)
#cv.resizeWindow("binario", 600, 800)

#cv.imshow("Original", img)
#cv.imshow("Erosion", erosion)

#diff = cv.absdiff(img, img2)
#cv.imshow("Diferenca", diff)

contornos, hierarquia = cv.findContours(
    binario,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)

# Pega o maior contorno, que deve ser o objeto
contorno = max(contornos, key=cv.contourArea)

x, y, largura, altura = cv.boundingRect(contorno)

print(f"x={x}, y={y}, largura={largura}, altura={altura}")

cv.rectangle(
    img,
    (x, y),
    (x + largura, y + altura),
    0,
    2
)
altura_da_imagem, largura_da_imagem = img.shape

x_centro = (x+largura/2) /largura_da_imagem
y_centro =(y+altura/2) /altura_da_imagem

altura_normalizada = altura/altura_da_imagem
largura_normalizada = largura/largura_da_imagem



#cv.imshow("binario", binario)

#cv.namedWindow("Bounding Box", cv.WINDOW_NORMAL)
#cv.resizeWindow("Bounding Box", 600, 800)

#cv.imshow("Bounding Box", img)
nome_foto = os.path.splitext(os.path.basename(img_caminho))[0]
print(nome_foto)

txt_box = open(f"{nome_foto}.txt", "w", encoding="utf-8")
txt_box.write(f"0 {x_centro} {y_centro} {largura_normalizada} {altura_normalizada}")
txt_box.close()

cv.waitKey(0)
cv.destroyAllWindows()