
### Intro

This is a project used opencv built-in library for face detection and recognition.

### Dependencies

Python 3.5.4, NumPy 1.12.1

### Getting started

0. I used OpenCV Contrib Version for this project.
  ```
  pip install opencv-contrib-python
  ```

1. Run detect_faces.py to make sure all required libraries are installed and able to use the webcam.
  ```
  python detect_faces.py
  ```

2. Run train_faces.py with name and the count of images you want to be trained in model. If you want to train multiple faces, you can run this script again with other person and label it with other name.
  ```
  python train_faces.py -n kitson -c 100
  ```

3. Run rec_faces.py to detect faces and label.
  ```
  python rec_faces.py
  ```

### To be continued
