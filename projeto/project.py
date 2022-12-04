import cv2
import matplotlib.pyplot as plt

# abrindo arquivos - modeloo pré-treinado para detecção de objetos
config_file = 'modelo_ia/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_modelo = 'modelo_ia/frozen_inference_graph.pb' 

# carregando modelo e configuração
modelo = cv2.dnn_DetectionModel(frozen_modelo, config_file)

lista_objetos = []
# carregando Labels
with open('modelo_ia/lista_objetos.txt', 'rt') as fpt:
    lista_objetos = fpt.read().rstrip('\n').split('\n')

# teste - carregamento correto da lista
#print(lista_objetos)
#print(len(lista_objetos))

# definindo tamanho suportado pelo modelo -> 320/320
modelo.setInputSize(320,320)
modelo.setInputScale(1.0/127.5)
modelo.setInputMean((127.5,127.5,127.5))
modelo.setInputSwapRB(True)

# IMAGEM
for i in range(9):
    img_name = 'img'
    img_name += str(i)
    img = cv2.imread("images_raw/" + img_name + ".jpg")

#fazendo a detecção dos objetos na imagem
# retorna -> classindex - indice do objeto na lista (exemplo -> cars == '3')
#         -> confidence - confiabilidade da detecção
#         -> bbox - posiçao e dimensões da caixa onde o objeto foi detectado
#Obs: configura a precisão modificando o Threshold
    ClassIndex, confidence, bbox = modelo.detect(img,confThreshold=0.56) #50% de precisão

#Destacando os objetos reconhecidos e salvando uma imagem com o resultado
    font_scale = 4
    font = cv2.FONT_HERSHEY_PLAIN
    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    # adicionando as caixas de destaques dos objetos
        cv2.rectangle(img, boxes, (255, 255, 0), 4)
    # adicionando os nomes dos objetos detectados dentro das respectivas caixas
        cv2.putText(img, lista_objetos[ClassInd-1], (boxes[0]+10, boxes [1]+40), font, fontScale=font_scale, color=(255, 255, 0), thickness = 2)
        
# salvando a imagem resultado
    cv2.imwrite("result/" + img_name + "_detected.jpg",img) 
    