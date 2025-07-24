import cv2 as cv
import numpy as np
from pyfirmata2 import Arduino

# Configuração do Arduino
porta = "COM6"  # Substitua pela aa correta do seu Arduino
arduino = Arduino(porta)
servoX = arduino.get_pin('d:3:s')  # Servo conectado no pino 3
servoY = arduino.get_pin('d:5:s')  # Servo conectado no pino 5

# d : pino digital / 7 ou 10 : número do pino  / s : confoguração para controle de servomotor


# Posições iniciais dos servos
current_servoX = 90  # Centralizado
current_servoY = 80
servoX.write(current_servoX)
servoY.write(current_servoY)

# Configuração do OpenCV
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():                      #Essa função retorna True se a câmera foi aberta com sucesso.
                                            #Retorna False se deu algum erro
    print("Câmera não pode ser acessada.")
    exit()

# Resolução da câmera
# frame_width = 640
# frame_height = 480
frame_width = 1024
frame_height = 1024
cap.set(3, frame_width) #configura a largura da imagem capturada (cv.CAP_PROP_FRAME_WIDTH)
cap.set(4, frame_height) #configura a altura da imagem capturada (cv.CAP_PROP_FRAME_HEIGHT)


# Função para mapear valores (similar ao map() do Arduino)
def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

#exemplo:
## Quer converter 512 (meio de 0 a 1024) para o intervalo de 0 a 180:
#map_value(512, 0, 1024, 0, 180)
#Resultado: ≈ 90.0
#Ou seja, o número 512 está no meio de 0 e 1024, então ele vira 90 (meio de 0 a 180).


# Zona morta para movimentos pequenos
dead_zone = 10  # Pixels : zona de tolarancia

# Variáveis para média móvel
history_size = 5  # Número de leituras para calcular a média
x_history = []   #para armazenar as últimas 5 posições horizontais (largura) do centro do rosto.
y_history = []   # para armazenar as últimas 5 posições verticais (altura) do centro do rosto.

# Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar a imagem.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))    #scaleFactor=1.1: Ele reduz o tamanho da imagem em 10% a cada passo para tentar detectar rostos de tamanhos diferentes.

    if len(faces) > 0:
        # Detecta o rosto principal (primeiro da lista)
        x, y, w, h = faces[0]    #primeira face
        face_center_x = x + w // 2
        face_center_y = y + h // 2     # // : divisão inteira

        # Adiciona a posição atual ao histórico
        x_history.append(face_center_x)
        y_history.append(face_center_y)

        # Mantém o histórico limitado
        if len(x_history) > history_size:
            x_history.pop(0)           #remove o elemnto mais antigo de largura
        if len(y_history) > history_size:
            y_history.pop(0)        #remove o elemnto mais antigo de altura

        # Calcula a média das posições para suavizar
        avg_x = sum(x_history) / len(x_history)
        avg_y = sum(y_history) / len(y_history)

        # Verifica se o movimento é maior que a zona morta
        delta_x = abs(avg_x - (frame_width / 2))   #módulo da diferença entre media x e centro da largura
        delta_y = abs(avg_y - (frame_height / 2)) #módulo da diferença entre media x e centro da largura
                                                #É a distância absoluta (sem sinal) do rosto em relação ao centro da tela, tanto na horizontal (x) quanto na vertical (y).
        if delta_x > dead_zone:
            target_servoX = map_value(avg_x, 0, frame_width, 180, 0)
            current_servoX = max(0, min(180, target_servoX))
            servoX.write(current_servoX)

        if delta_y > dead_zone:
            target_servoY = map_value(avg_y, 0, frame_height, 80, 160)
            current_servoY = max(80, min(160, target_servoY))
            servoY.write(current_servoY)

        # Desenha o círculo no rosto detectado
        radius = max(w, h) // 3  # Define o raio do círculo com base no tamanho do rosto : valor testado 
        cv.circle(frame, (face_center_x, face_center_y), radius, (0, 255, 0), 2)  # Círculo

        # Desenha as linhas da mira
        cv.line(frame, (face_center_x, 0), (face_center_x, frame_height), (0, 255, 0), 2)  # Linha vertical
        cv.line(frame, (0, face_center_y), (frame_width, face_center_y), (0, 255, 0), 2)  # Linha horizontal

        # Desenha a bolinha vermelha no centro
        cv.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)  # Bolinha vermelha no centro

        # Mostra as coordenadas na tela
        cv.putText(frame, f"Servo X: {int(current_servoX)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.putText(frame, f"Servo Y: {int(current_servoY)}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Exibe o vídeo com a detecção de rosto
    cv.imshow("Face Tracking", frame)

    # Encerra com a tecla 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Finaliza o programa
cap.release()
cv.destroyAllWindows()
