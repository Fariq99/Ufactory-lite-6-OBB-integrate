from ultralytics import YOLO
import math

def run_detection(target_classes):
    # Load the YOLO OBB model
    model = YOLO('cobotCvV2.pt')  # Assuming you're using YOLOv8 with OBB

    # Run inference on the webcam input (source=1 for webcam)
    results = model("WIN_20241021_10_08_39_Pro.jpg")

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
            
            # Filter detections based on confidence score and class name
            if confidence >= 0.6 and det_name in target_classes:
                x = obb.xywhr[i][0] 
                y = obb.xywhr[i][1]
                angle = obb.xywhr[i][4]
                angle_degrees = angle * (180 / math.pi)
                print(f"Detected {det_name} with confidence {confidence:.2f} at coordinates (x={x}, y={y}, angle(degree)={angle_degrees})")
    else:
        print("No OBB detections were made.")

# Accept target classes (filter by "spoon", "toothbrush", "fork")
target_input = input("Enter target objects (comma separated): ")
target_classes = [item.strip().lower() for item in target_input.split(",")]

# Run the detection for the given target objects
run_detection(target_classes)
