
### Intro

This is a project used opencv built-in library for face detection and recognition.

### Tested Environment

macOS High Sierra 10.13.1
Ubuntu 16.04

### Dependencies

Python 3.5.4, NumPy 1.12.1

### Getting started

0. Download haarcascades_frontalface_default.xml from [here](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

1. I used OpenCV Contrib Version for this project.
  ```
  pip install opencv-contrib-python
  ```

2. Run detect_faces.py to make sure all required libraries are installed and able to use the webcam.
  ```
  python detect_faces.py
  ```

3. Run train_faces.py with name and the count of images you want to be trained in model. If you want to train multiple faces, you can run this script again with other person and label it with other name.
  ```
  python train_faces.py -n kitson -c 100
  ```

4. Run rec_faces.py to detect faces and label.
  ```
  python rec_faces.py
  ```

### To be continued
