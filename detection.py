import dlib
from imutils import face_utils
import numpy as np
import cv2

video_capture = cv2.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()
predictor_path = "../DlibData/shape_predictor_68_face_landmarks.dat"
face_predictor = dlib.shape_predictor(predictor_path)

while True:
    ret, img = video_capture.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    for face in faces:
        landmark = face_predictor(gray, face)
        landmark = face_utils.shape_to_np(landmark)

        for (i, (x, y)) in enumerate(landmark):
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("image", img)

    if cv2.waitKey(1) == ord('q'):
        break
