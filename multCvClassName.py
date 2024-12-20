from ultralytics import YOLO

# Load a model
model = YOLO("cobotCvV2.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model("WIN_20241021_10_08_39_Pro.jpg",conf=0.6,save=True) 

for result in results:
    obb = results[0].obb
    names = results[0].names
    for i in range(len(obb.cls)):  # Loop over all detections
        class_id = obb.cls[i]  # Class ID of the i-th detection
        confidence = obb.conf[i]  # Confidence score of the i-th detection
        det_name = names[int(class_id)]
        print(f"Object {i + 1}: Class ID = {class_id}, Class name = {det_name}, Confidence = {confidence}")
