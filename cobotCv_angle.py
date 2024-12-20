import sys
import math
import time
import queue
import datetime
import random
import traceback
import threading
import cv2
from xarm import version
from xarm.wrapper import XArmAPI
from ultralytics import YOLO

angle_degrees = None
delta_x = None
delta_y = None
adjust_x = None
adjust_y = None

def capture_image_from_webcam(image_path="captured_image.jpg", width=1920, height=1080):
    # Initialize webcam
    cap = cv2.VideoCapture(0) # 0 is the default webcam

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
    global angle_degrees
    global adjust_x
    global adjust_y

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
                angle_degrees = (angle * (180 / math.pi)) + 90
                adjust_x = x * 0.182
                adjust_y = y * 0.185
                print(f"Detected {det_name} with confidence {confidence:.2f} at coordinates (x={x}, y={y}, angle(degree)={angle_degrees})")
    
    else:
        print("No OBB detections were made.")

class RobotMain(object):
    """Robot Main Class"""
    def __init__(self, robot, **kwargs):
        self.alive = True
        self._arm = robot
        self._ignore_exit_state = False
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self._vars = {}
        self._funcs = {}
        self._robot_init()

    # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if not self._ignore_exit_state and data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code, self._arm.connected, self._arm.state, self._arm.error_code, ret1, ret2))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1], ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._ignore_exit_state:
                return True
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    # Robot Main Run
    def run(self):
        try:
            code = self._arm.set_servo_angle(angle=[0, -15, 45, 0, 60, 0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_tool_position(*[180, -174.72, 0.0, 0.0, 0.0, 0.0], speed=self._tcp_speed, mvacc=self._tcp_acc, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_tool_position(*[0.0, 0.0, 0.0, 0.0, 0.0, 90], speed=self._tcp_speed, mvacc=self._tcp_acc, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            self._vars['x'] = adjust_x
            self._vars['y'] = adjust_y
            code = self._arm.set_tool_position(*[self._vars.get('x', 0),self._vars.get('y', 0), 0.0, 0.0, 0.0, 0.0], speed=self._tcp_speed, mvacc=self._tcp_acc, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            self._vars['obb_angle'] = angle_degrees
            code = self._arm.set_tool_position(*[0.0, 0.0, 0.0, 0.0, 0.0, self._vars.get('obb_angle', 0)], speed=self._tcp_speed, mvacc=self._tcp_acc, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            code = self._arm.set_position(z=-232.7, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.close_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            code = self._arm.set_position(z=232.7, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(z=-232.7, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            code = self._arm.set_position(z=232.7, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.close_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
            code = self._arm.set_servo_angle(angle=[0, -15, 45, 0, 60, 0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        finally:
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)


if __name__ == '__main__':
    image_path = capture_image_from_webcam(width=1920,height=1080)
    target_input = input("Enter target objects (comma separated): ")
    target_classes = [item.strip().lower() for item in target_input.split(",")]
    if image_path:
        run_detection(image_path, target_classes)
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.4.224', baud_checkset=False)
    robot_main = RobotMain(arm)
    robot_main.run()
    print(angle_degrees)
    print(adjust_x)
    print(adjust_y)
