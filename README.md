
# Object Detection

Projeto de C209

Esse projeto consiste no uso de uma inteligência artificial pré-treinada para detecção de vários obejtos em imagens.



## Uso/Exemplos

- Isso irá salvar uma imagem "image_name_detected.jpg" com os objetos identificados em destaque, na pasta 'Result'.
```python
img_name = 'image'
img = cv2.imread(img_name + ".jpg")
object_detector(img)
```
![IMAGEM](https://github.com/Boremp/object_detection/blob/main/projeto/images_raw/img5.jpg)
![IMAGEM_RESULTADO](https://github.com/Boremp/object_detection/blob/main/projeto/result/img5_detected.jpg)

- Utilizando esse código pode-se adicionar todas as imagens que deseja-se fazer a avaliação na pasta "/images_raw" com o formato "imgNUM.jpg"
| /images_raw   :   | img0.jpg       |img1.jpg       | img2.jpg       | img3.jpg       |
| ----------------- |----------------| --------------| -------------- | -------------- |

```python
# Detecção de todas as imagens na pasta 'images_raw'
for i in range(n_img):
    img_name = 'img'
    img_name += str(i)
    img = cv2.imread("images_raw/" + img_name + ".jpg")
    object_detector(img)
```
