import cv2
import numpy as np
import os

def get_name(id):
    directory = os.listdir("./data")
    return directory[id]

def main():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    camera = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("model.yml")

    print("Person trained: ")
    directory = os.listdir("./data")
    for i in range(len(directory)):
        if directory[i] != ".DS_Store":
            print(str(i) + " " + directory[i])

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        frame = camera.read()[1]
        frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)

        for (x, y, w, h) in faces:
            p, con = recognizer.predict(gray[y : y + h, x : x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if con < 50:
                name = get_name(p)
                cv2.putText(frame, name, (x, y + h), font, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2,destroyAllWindows()

if __name__ == "__main__":
    main()
