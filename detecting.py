import cv2 as cv
import numpy as np

# Load the NumPy arrays with face data (with and without masks)
with_mask = np.load('./with mask.npy')
without_mask = np.load('./without mask.npy')

# Print the shapes of the arrays before reshaping
print("With mask shape before reshaping: ", with_mask.shape)
print("Without mask shape before reshaping: ", without_mask.shape)

# Reshape the arrays into a 2D format (200 samples of 50x50x3)
with_mask = with_mask.reshape(200, 50 * 50 * 3)
without_mask = without_mask.reshape(200, 50 * 50 * 3)

# Print the shapes of the arrays after reshaping
print("With mask shape after reshaping: ", with_mask.shape)
print("Without mask shape after reshaping: ", without_mask.shape)

# Concatenate the two arrays into a single feature matrix (X)
X = np.r_[with_mask, without_mask]

# Create labels (0 for with mask, 1 for without mask) for the samples
labels = np.zeros(X.shape[0])
labels[200:] = 1.0

# Create a dictionary to map labels to class names
names = {0: 'Mask', 1: 'No Mask'}

# Import necessary libraries for machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

# Print the shape of the training data
print("X train shape: ", x_train.shape)

# Perform dimensionality reduction using PCA
from sklearn.decomposition import PCA 
pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)

# Print the shape of the reduced training data
print("X_train new Shape: ", x_train.shape)

# Create a Support Vector Machine (SVM) classifier
svm = SVC()
svm.fit(x_train, y_train)

# Transform the test data with PCA and make predictions
x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)

# Print the accuracy of the model
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Load a Haar Cascade classifier for face detection
haar_data = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

# Start the camera capture
capture = cv.VideoCapture(0)
data = []
font = cv.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv.resize(face, (50, 50))
            face = face.reshape(1, -1)
            face = pca.transform(face)  # Apply PCA to match the dimensionality
            prediction = svm.predict(face)
            text = " "
            cv.putText(img, text, (x, y), font, 1, (244, 250, 250), 2)
            if prediction == 1:
                text = "No Mask"
                print(text)
            else:
                text = "Mask"
                print(text)
        cv.imshow('result', img)
        if cv.waitKey(2) == 27:
            break

# Release the camera and close OpenCV windows
capture.release()
cv.destroyAllWindows()
