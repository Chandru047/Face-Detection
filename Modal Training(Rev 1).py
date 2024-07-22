import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping
from keras.src.regularizers import L2
import pickle


# Parameters
path = "datasets"  # Folder with all the class folders
labelFile = "labels.csv"  # File with all names of classes
batch_size_val = 10  # How many to process together
steps_per_epoch_val = 100
epochs_val = 50
imageDimensions = (64, 64, 1)  # Use the same dimensions as your captured images with grayscale (1 channel)
testRatio = 0.8   # If 1000 images, split will be 200 for testing
validationRatio = 0.2 # If 1000 images, 20% of remaining 800 will be 160 for validation

# Importing Images and Preprocessing
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for class_dir in myList:
    if not os.path.isdir(os.path.join(path, class_dir)):
        continue
    myPicList = os.listdir(os.path.join(path, class_dir))
    for y in myPicList:
        curImg = cv2.imread(os.path.join(path, class_dir, y))
        curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))  # Resize image
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

# Print the shape of the imported images to debug
print(f"Shape of imported images: {images.shape}")
print(f"Expected shape: (-1, {imageDimensions[0]}, {imageDimensions[1]})")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# Ensure the shapes are correct
print("Data Shapes")
print("Train", end=""); print(X_train.shape, y_train.shape)
print("Validation", end=""); print(X_validation.shape, y_validation.shape)
print("Test", end=""); print(X_test.shape, y_test.shape)
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels in training set"
assert(X_validation.shape[0] == y_validation.shape[0]), "The number of images is not equal to the number of labels in validation set"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels in test set"
assert(X_train.shape[1:] == imageDimensions[:2]), "The dimensions of the Training images are wrong"
assert(X_validation.shape[1:] == imageDimensions[:2]), "The dimensions of the Validation images are wrong"
assert(X_test.shape[1:] == imageDimensions[:2]), "The dimensions of the Test images are wrong"

# Read CSV File
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

# Display Some Samples Images of All the Classes
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1)], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

# Display a Bar Chart Showing No of Samples for Each Category
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# Preprocessing the Images
def grayscale(img):
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocess_and_reshape(img):
    img = grayscale(img)  # Convert to grayscale
    img = equalize(img)    # Standardize the lighting in an image
    img = img / 255        # Normalize values between 0 and 1 instead of 0 to 255
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale images
    return img

X_train = np.array(list(map(preprocess_and_reshape, X_train)))  # Iterate and preprocess all images
X_validation = np.array(list(map(preprocess_and_reshape, X_validation)))
X_test = np.array(list(map(preprocess_and_reshape, X_test)))
cv2.imshow("GrayScale Images", X_train[random.randint(0, len(X_train) - 1)])  # Check if the training is done properly

# Augmentation of Images: To Make It More Generic
dataGen = ImageDataGenerator(
    width_shift_range=0.1,  # 0.1 = 10% if more than 1 e.g 10 then it refers to no. of pixels e.g 10 pixels
    height_shift_range=0.1,
    zoom_range=0.2,  # 0.2 means can go from 0.8 to 1.2
    shear_range=0.1,  # Magnitude of shear angle
    rotation_range=10,  # Degrees
    horizontal_flip=True,  # Enable horizontal flipping
    fill_mode='nearest'  # Fill mode for filling in newly created pixels
)
dataGen.fit(X_train)

# Convolution Neural Network Model
def myModel():
    no_Of_Filters = 35
    size_of_Filter = (3, 3)  # This is the kernel that moves around the image to get the features.
    size_of_Filter2 = (3, 3)
    size_of_pool = (3, 3)  # Scale down all feature maps to generalize more, to reduce overfitting
    no_Of_Nodes = 300  # No. of nodes in hidden layers
    model = Sequential()
    model.add(Input(shape=(imageDimensions[0], imageDimensions[1], 1)))

    # Add L2 regularization to Conv2D layers
    model.add(Conv2D(no_Of_Filters, size_of_Filter, activation='relu', kernel_regularizer=L2(0.01)))
    model.add(Conv2D(no_Of_Filters, size_of_Filter, activation='relu', kernel_regularizer=L2(0.01)))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu', kernel_regularizer=L2(0.01)))
    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu', kernel_regularizer=L2(0.01)))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())

    # Add L2 regularization to Dense layers
    model.add(Dense(no_Of_Nodes, activation='relu', kernel_regularizer=L2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    # Compile model
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train Model
model = myModel()
history = model.fit(
    dataGen.flow(X_train, to_categorical(y_train, noOfClasses), batch_size=batch_size_val),
    steps_per_epoch=steps_per_epoch_val,
    epochs=epochs_val,
    validation_data=(X_validation, to_categorical(y_validation, noOfClasses)),
    callbacks=[early_stopping],
    shuffle=True
)

# Plot training and validation loss and accuracy
plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate on test set
score = model.evaluate(X_test, to_categorical(y_test, noOfClasses), verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# Store the model as a pickle object

pickle_out = open("model_trained.p", "wb")  # wb = Write Byte
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)
