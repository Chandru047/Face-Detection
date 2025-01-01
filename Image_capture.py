import os
import cv2
import threading
from queue import Queue

# Parameters
dataset_dir = 'datasets'
frame_width = 320  # Smaller frame width for faster processing
frame_height = 240  # Smaller frame height for faster processing
image_size = (64, 64)  # Resize images to this size
num_images_per_class = 1000  # Number of images to capture per class

# Create directories for datasets
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError("Failed to load Haar Cascade Classifier xml file.")

# Function to detect and save face
def detect_and_save_face(frame, label, count):
    # Convert the frame to grayscale (needed for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y + h, x:x + w]

        # Save the face ROI
        label_dir = os.path.join(dataset_dir, label)
        frame_filename = os.path.join(label_dir, f'{label}_{count}.jpg')
        cv2.imwrite(frame_filename, face_roi)
        return True
    return False

# Worker function to process frames from the queue
def worker(label, count, q):
    while True:
        frame = q.get()
        if frame is None:
            break
        if detect_and_save_face(frame, label, count[0]):
            count[0] += 1
        q.task_done()

# Function to capture images for a specific label
def capture_images(label):
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    count = [0]
    frame_queue = Queue()

    # Start a worker thread
    thread = threading.Thread(target=worker, args=(label, count, frame_queue))
    thread.start()

    # Create a single window for displaying the video feed
    cv2.namedWindow(f'Capturing {label}', cv2.WINDOW_NORMAL)

    while count[0] < num_images_per_class:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue

        # Display the live video
        cv2.imshow(f'Capturing {label}', frame)

        # Add frame to the queue
        frame_queue.put(frame)

        # Exit when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Capture interrupted by user.")
            break
        elif key == ord('c'):  # Press 'c' to capture an image
            if detect_and_save_face(frame, label, count[0]):
                count[0] += 1

    # Stop the worker thread
    frame_queue.put(None)
    thread.join()

    cap.release()
    cv2.destroyAllWindows()
# Main function to execute the image capture process
def main():
    while True:
        label = f'class_{len(os.listdir(dataset_dir))}'  # Automatically generate label
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        print(f'Capturing images for {label}...')
        capture_images(label)
        print(f'Finished capturing {num_images_per_class} images for {label}.')

        if len(os.listdir(label_dir)) < num_images_per_class:
            print(f"Warning: {label} has less than {num_images_per_class} images captured.")

        choice = input("Do you want to capture images for another class? (yes/no): ").lower()
        if choice != 'yes':
            break

if __name__ == "__main__":
    main()
