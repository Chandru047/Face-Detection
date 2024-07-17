import cv2
import pickle
import numpy as np

# SETUP THE VIDEO CAMERA
frameWidth = 320  # CAMERA RESOLUTION
frameHeight = 240
brightness = 180
font = cv2.FONT_HERSHEY_SIMPLEX

# SET UP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRAINED MODEL USING PICKLE
try:
    with open('model_trained.p', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model file not found. Please ensure 'model_trained.p' exists.")
    exit()

# LOAD HAAR CASCADE FOR FACE DETECTION
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize(img):
    return cv2.equalizeHist(img)


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    img = cv2.resize(img, (64, 64))  # Adjust to match training input dimensions
    img = img.reshape(1, 64, 64, 1)
    return img


def getClassName(classNo):
    class_names = {
        0: 'chandru',
        1: 'krishna',
        # Add more class mappings as needed
    }
    return class_names.get(classNo, 'Unknown')


def check_for_person(face_img, threshold=0.7, unknown_threshold=0.4):
    img = preprocessing(face_img)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        return classIndex, probabilityValue
    elif probabilityValue > unknown_threshold:
        return None, probabilityValue  # Adjust this to return 'Unknown' or None as needed
    else:
        return None, None


while True:
    # READ IMAGE
    success, imgOriginal = cap.read()
    if not success:
        print("Failed to read from camera. Exiting...")
        break

    # DETECT FACES
    gray = grayscale(imgOriginal)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    face_data = []

    for (x, y, w, h) in faces:
        face_img = imgOriginal[y:y + h, x:x + w]
        classIndex, probabilityValue = check_for_person(face_img, threshold=0.95, unknown_threshold=0.5)  # Adjust thresholds as needed
        if classIndex is not None:
            face_data.append((x, y, w, h, classIndex, probabilityValue))

    # Find the face with the highest probability for each class
    class_max_prob = {}
    for data in face_data:
        x, y, w, h, classIndex, probabilityValue = data
        if classIndex not in class_max_prob or probabilityValue > class_max_prob[classIndex][1]:
            class_max_prob[classIndex] = (data, probabilityValue)

    for data in face_data:
        x, y, w, h, classIndex, probabilityValue = data
        if (x, y, w, h, classIndex, probabilityValue) == class_max_prob[classIndex][0]:
            detected_person = getClassName(classIndex)
        else:
            detected_person = 'Unknown'

        # Draw a rectangle around the face and display the name
        cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(imgOriginal, detected_person, (x, y - 10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
