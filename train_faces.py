import cv2
import argparse
import sys
import os
import numpy as np
from PIL import Image

def face_capture(i, num):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    camera = cv2.VideoCapture(0)
    n = 0

    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("./data/" + i):
        os.makedirs("data/" + i)
    # else:
    #     print("Person trained")
    #     sys.exit(-1)

    while True:
        frame = camera.read()[1]

        frame = cv2.resize(frame, (640, 360))

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)

        for (x, y, w, h) in faces:
            n += 1
            face_to_save = gray[y : y + h, x : x + w]
            cv2.imwrite("./data/" + i + "/" + str(n) + ".jpg", face_to_save)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif n == int(num):
            break

    camera.release()
    cv2.destroyAllWindows()

def face_train():
    faceSamples = []
    Ids = []

    directory = os.listdir("./data")
    len_dir = len(directory)

    print("Person to train:")

    for i in range(len(directory)):
        if directory[i] != ".DS_Store":
            print(str(i) + " " + directory[i])
            for j in os.listdir("./data/" + directory[i]):
                if j != ".DS_Store":
                    pilImage = Image.open("./data/" + directory[i] + "/" + j).convert('L')
                    imageNp = np.array(pilImage, 'uint8')
                    faceSamples.append(imageNp)
                    Ids.append(i)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faceSamples, np.array(Ids))
    recognizer.write("model.yml")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help = "name")
    parser.add_argument('-c', '--count', help = "count")
    args = parser.parse_args()
    print("Program starts")
    face_capture(args.name, args.count)
    face_train()
    print("Trainning completed. Run rec_faces.py")

if __name__ == "__main__":
    main()
