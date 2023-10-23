import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv.imread("./travis-kelce-jason-kelce.jpg")
new_width = 1200
new_height = 700

# Resize the image
resized_img = cv.resize(img, (new_width, new_height))

# Load a Haar Cascade classifier for face detection
haar_data = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

# Uncomment this block if you want to detect faces in the resized image
# while True:
#     faces = haar_data.detectMultiScale(resized_img)
#     for x, y, w, h in faces:
#         cv.rectangle(resized_img, (x, y), (x + w, y + h), (255, 0, 255), 4)
#     cv.imshow('result', resized_img)
#     # 27 - ASCII Code for Escape
#     if cv.waitKey(2) == 27:
#         break
# cv.destroyAllWindows()

# Start the camera capture
capture = cv.VideoCapture(0)
data = []

while True:
    flag, img = capture.read()  # Flag to indicate whether the camera is open or closed

    if flag:
        # Detect faces in the captured frame
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv.resize(face, (50, 50))
            print(len(data))
            if len(data) < 400:
                data.append(face)
        cv.imshow('result', img)
        # 27 - ASCII Code for Escape or stop when you have collected 200 samples
        if cv.waitKey(2) == 27 or len(data) >= 200:
            break

# Save the collected data (face images with masks) to a NumPy file
np.save("with_mask.npy", data)

# Display the first collected face image
plt.imshow(data[0])

# Release the camera and close OpenCV windows
capture.release()
cv.destroyAllWindows()
