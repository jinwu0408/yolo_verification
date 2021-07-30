
# General Description

This project is intended to send a drone to verify/enhance the yolov5 object detection with a low confidence score.


## Project Breakdown
- Train yolov5 for the custom object.(yolo_model/train.py)
- Run yolov5 on webcam. (main2.py)
  - If confidence above threshold:
    - Continue(main2.py)
  - If confidence below threshold:
    - Figure out where to send the drone to **(Need to Do)**
    - Connect to the drone. (collect_data.py)
    - Sent the drone **(Need to do)**
    - Collect save footage from the drone. (collect_data.py)
    - Save the footages and disconnect the drone.
 (collect_data.py)
    - Run yolov5 on collected footage, and update the score with the maximum confidence score.(update_conf_score.py)

## Usage
- run main2.py
- One window will pop up showing webcam
- It will continue run detection, once there is detection with confidence below it will ask for 'Low confidence score, send a drone?'
  - Yes
    - it will all keyboard_ctrl and press 0 to take photos from the drone.
  - No
    - It will continue running.

## KeyboardCtrl
    Ctrl.QUIT: Key.esc,\
    Ctrl.TAKEOFF: "t",\
    Ctrl.LANDING: "l",\
    Ctrl.MOVE_LEFT: "a",\
    Ctrl.MOVE_RIGHT: "d",\
    Ctrl.MOVE_FORWARD: "w",\
    Ctrl.MOVE_BACKWARD: "s",\
    Ctrl.MOVE_UP: Key.up,\
    Ctrl.MOVE_DOWN: Key.down,\
    Ctrl.TURN_LEFT: Key.left,\
    Ctrl.TURN_RIGHT: Key.right,\
    Ctrl.BREAK_LOOP: Key.enter,\
    Ctrl.WAIT: Key.space,\
    Ctrl.CHECK: "c",\
    Ctrl.Streaming_START:"1",\
    Ctrl.Streaming_STOP:"2",\
    Ctrl.RECORDING_START:"3",\
    Ctrl.RECORDING_STOP:"4", \
    Ctrl.TAKE_PHOTO:"5"\
