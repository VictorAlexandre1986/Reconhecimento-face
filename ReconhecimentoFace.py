#pip install numpy
#pip install opencv
#pip install opencv_contrib_python

import numpy as np
import cv2
import os

#Função savePerson()
def savePerson():
    global identificacao
    global boolsavingimg
    print('Qual seu nome ?')
    name = input()
    identificacao = name
    boolsavingimg = True

#Função saveImg
def diretorio(img):
    global identificacao

    if not os.path.exists('train'):
        #os.makedirs(f'train/{identificacao}',mode=0o777, exist_ok=True)
        os.makedirs(f'train/', mode=0o777, exist_ok=True)
    os.makedirs(f'train/{identificacao}', mode=0o777, exist_ok=True)
    #Cria arquivos na pasta
    files = os.listdir(f'train/{identificacao}')
    cv2.imwrite(f'train/{identificacao}/{str(len(files))}.jpg',img)

def treinamentoModelo():
    global recognizer
    global trained
    trained= True
    persons = os.listdir('train')
    ids=[]
    faces=[]
    for i, p in enumerate(persons):
        for f in os.listdir(f'train/{p}'):
            img = cv2.imread(f'train/{p}/{f}', 0)
            faces.append(img)
            ids.append(i)
    recognizer.train(faces, np.array(ids))
    print('Treino finalizado!')

#Variáveis
identificao=''
boolsavingimg = False
saveCount = 0
trained = False
persons = os.listdir('train')

#Reconhecedor
#Precisa instalar o LBPH, pip install opencv_contrib_python
recognizer = cv2.face.LBPHFaceRecognizer_create()

#Ler Webcam
cap = cv2.VideoCapture(0)

#Carregar o xml do Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')

#loop
while(True):

    #Webcam para frame
    ret, frame = cap.read()

    #Frame em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detecção da  face no frame 'gray'
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Iterar todas as faces no frame
    for (x, y, w, h) in faces:

        #Cortar face
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (50,50))

        #Colocar o retângulo na face do frame
        cv2.rectangle(frame, (x,y), (x+w, y+h), (200,0,0), 2)

        #Teste
        if trained:
            idp, acc = recognizer.predict(roi)
            namePerson = persons[idp]
            cv2.putText(frame, namePerson, (x,y), cv2.FONT_HERSHEY_SIMPLEX ,2, (0,255,0), 2, cv2.LINE_AA)


        #Checa boolsavingimg
        if boolsavingimg:
            diretorio(roi)
            saveCount +=1


        if saveCount > 100:
            boolsavingimg = False
            saveCount=0

    #Exibir o frame
    cv2.imshow('frame',frame)

    #Recupera botão
    key = cv2.waitKey(1)

    #Close loop
    if key == ord('c'):
        break

    #Salvar imagens
    if key == ord('s'):
        savePerson()

    #Treinamento do modelo
    if key == ord('t'):
        treinamentoModelo()


#Fechando a janela e reinicializando a mesma
cap.release()
cv2.destroyAllWindows()