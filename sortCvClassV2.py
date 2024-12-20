import cv2
from ultralytics import YOLO
import math

def capture_image_from_webcam(image_path="captured_image.jpg"):
    # Initialize webcam
    cap = cv2.VideoCapture(1)  # 0 is the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    # Capture frame
    ret, frame = cap.read()

    # Release the webcam
    cap.release()

    if not ret:
        print("Error: Could not read frame from webcam.")
        return None

    # Save the captured frame to a file
    cv2.imwrite(image_path, frame)
    print(f"Image saved as {image_path}")

    return image_path

def run_detection(image_path, target_classes):
    # Load the YOLO OBB model
    model = YOLO('cobotCvV2.pt')

    # Run inference on the saved image
    results = model(source=image_path)  # Image source

    # Access the OBB detections
    obb = results[0].obb  # Results for the current frame/image
    names = results[0].names  # Dictionary mapping class IDs to class names

    # Ensure there are OBB detections
    if obb is not None and len(obb.cls) > 0:
        # Iterate over each detected object
        for i in range(len(obb.cls)):
            # Get the class name and confidence for each detected object
            det_class_id = obb.cls[i]  # Class ID of the i-th detection
            det_name = names[int(det_class_id)]  # Class name
            confidence = obb.conf[i]  # Confidence score of the i-th detection
            
            # Filter detections based on class name and confidence threshold
            if det_name in target_classes and confidence >= 0.6:
                # Get the x, y coordinates from the obb.xywhr (x_center, y_center, width, height, rotation)
                x = obb.xywhr[i][0]  # x_center
                y = obb.xywhr[i][1]  # y_center
                angle = obb.xywhr[i][4]
                angle_degrees = angle * (180 / math.pi)
                print(f"Detected {det_name} with confidence {confidence:.2f} at coordinates (x={x}, y={y}, angle(degree)={angle_degrees})")
    else:
        print("No OBB detections were made.")

# Capture an image from the webcam
image_path = capture_image_from_webcam()

# Accept target classes (filter by "spoon", "toothbrush", "fork")
target_input = input("Enter target objects (comma separated): ")
target_classes = [item.strip().lower() for item in target_input.split(",")]

# If an image was successfully captured, run detection
if image_path:
    run_detection(image_path, target_classes)
