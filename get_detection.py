import cv2
import numpy as np
import os

path_annotations = 'urna/annotations/'

# Verifica se a pasta existe
if not os.path.exists(path_annotations):
    # Cria a pasta se ela não existir
    os.makedirs(path_annotations)

for i in range(119):
    # Ler a imagem e a máscara segmentada
    img = cv2.imread('urna/images/'+str(i)+'.jpg')
    mask = cv2.imread('urna/masks/'+str(i)+'.png', 0)

    # Aplica um filtro de suavização
    kernel_size = 5
    blur = cv2.blur(mask, (kernel_size, kernel_size))

    # Aplicar uma operação morfológica para remover ruídos e fechar buracos na máscara
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Encontrar o contorno do objeto na máscara
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar o contorno com a maior área
    max_contour = max(contours, key=cv2.contourArea)

    # Desenhar o contorno com a maior área na imagem original
    cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 3)

    # Obter o retângulo delimitador ao redor do objeto
    x, y, w, h = cv2.boundingRect(max_contour)

    # Desenhar o retângulo delimitador na imagem original
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Salvar anotações
    with open(path_annotations + str(i) + '.txt', 'w') as f:
        f.write(str(x)+','+str(y)+','+str(x+w)+','+str(y+h))

