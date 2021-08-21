
# General Description

This project is intended to send a drone to verify/enhance the yolov5 object detection with a low confidence score.


## Usage
- Train yolov5 for the custom object.(yolo_model/train.py)
- Drag the trained weight to the root folder.
- In detect.py, change the weights that the Yolo model will use.
- Start the API Server use API.apiserver.py
```
python apiserver.py
```
- In a new terminal, run main2.py.
```
python main2.py
```
- Run yolov5 on webcam. (main2.py)
  - The frames with detection will save to frame_dir/primary/, and upload to the database
  - If confidence above threshold:
    - Continue(main2.py)
  - If confidence below threshold:
    - Add tasks to the drone(Taking off, move, take pics, land)(collect_data.py)
    - Analyze the new data(saved in frame_dir/secondary/).(Update_conf_score.py)
    - Update the new confidence and the actual label in the database. (Update_conf_score.py)

## API functions(In API2 folder)
- API2.clear_db:
    - Clear all data with the drone id in the database.

   - Parameters:
        - id (int): The drone id

   - Returns:
        - (String): "Cleared Successfully"
- API2.add_task:
    - Add the tasks using API

    - Parameters:
        - drone_id (int): The drone id
        - taskname(String): Task Name
        - task(String): Task parameters

    - Returns:
       -  (String): "Updated Successfully"
- API2.upload_img:
    - Upload the image into the database using the API.

    - Parameters:
        - path (String): The path of the image
        - id(int): The drone id.
        - x1(float): The center location.
        - x2(float): The center location.
        - y1(float): The center location.
        - y2(float): The center location.
        - label(String): The label for the detections.
        - confidence(float): The confidence for the detection.

    - Returns:
       - (String): 'Uploaded Successfully'
- API2.update_conf:
    - Update the confidence score and label of the image associates with the path
    in the database.

    - Parameters:
        - path (String): The path of the image
        - confidence(float): The confidence cutoff for yolo to determine
        - detections.
        - label(String): The label for the detections

    - Returns:
        - (String): 'Updated Successfully'

- API2.get_size_with_id:
    - Get the size of data stored for drone id.

    - Parameters:
      -   id (int): The drone id.

    - Returns:
       -  size(int): The size of data stored for drone id.


## KeyboardCtrl.py
- Control keys and their usage

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

## Tester Code and Scribbles
- landing.py: Land the drone immediately: Used as emergency shut down for the Anafi drone.
- move_in_square.py: Let the drone move in a square.
- streaming_example.py: Start and stop streaming on Anafi Drone.
- testing_streaming.py: Tester of the above code.
- tracker.py: Track the drone, returns the trace with a background in a plot.(Static environment)
