from API2.add_task import add_task
import time
# from API2.callget_drone_photo import setup_photo_burst_mode,take_photo_burst

#source ~/code/parrot-groundsdk/./products/olympe/linux/env/shell

def collect_data(drone_id):

    '''
    Coommunicate and add tasks for drone with the drone_id using the API.

    Parameters:
        drone_id (int): The Drone id.

    Returns:
        String: 'Fininshing Collecting the images'
    '''
    add_task(drone_id, 'TakeOff')
    add_task(drone_id, 'sleep', '10')
    add_task(drone_id, 'takepicture')
    add_task(drone_id, 'Landing')
    time.sleep(0.5)
    return 'Fininshing Collecting the images'
    # ret_list = []
    # cap = cv2.VideoCapture(0)
    # save_dir = 'frame_dir/secondary/'
    # ret_list = []
    # dirs = os.listdir( save_dir )
    #
    # for img in dirs:
    #    image = cv2.imread(save_dir+img)
    #    ret_list.append(image)
    # return ret_list




    # ctrl_seq = []
    # drone_ip ="192.168.42.1"# "10.202.0.1"  "192.168.42.1"
    # with olympe.Drone(drone_ip) as drone:
    #     print('Connecting the drone')
    #     drone.connection()
    #     drone(set_camera_mode(cam_id=1, value='photo')).wait()
    #     # setup_photo_mode(drone)
    #     drone.start_piloting()
    #     control = KeyboardCtrl()
    #
    #
    #     while not control.quit():
    #         if control.takeoff():
    #             drone(TakeOff())
    #         elif control.landing():
    #             drone(Landing())
    #         # elif control.take_photo():
    #             # streaming_example.save_frame()
    #         elif control.start_streaming():
    #             print('Starting streaming')
    #             streaming_example = StreamingExample()
    #             streaming_example.start()
    #         elif control.stop_streaming():
    #             print('Stop streaming')
    #             streaming_example.stop()
    #         elif control.has_piloting_cmd() or control.wait():
    #             if control.has_piloting_cmd():
    #                 ctrl_seq.append([control.roll(), control.pitch(), control.yaw(), control.throttle(), 0])  # 0 stands for not hover
    #                 drone.piloting_pcmd(control.roll(), control.pitch(), control.yaw(), control.throttle(), 0.02)
    #             else:  # when fly over the area needs to be clean
    #                 ctrl_seq.append([control.roll(), control.pitch(), control.yaw(), control.throttle(), 1])  # 1 stands for hover
    #                 drone.piloting_pcmd(0, 0, 0, 0, 0.02)
    #
    #         time.sleep(0.02)
    #     # print('Disconnecting the drone')
    #     # drone.disconnect()
    #     img_path = 'tmp_img_dir/'
    #     ret_list = []
    #     dirs = os.listdir( img_path )
    #
    #     for img in dirs:
    #        image = cv2.imread(img_path+img)
    #        ret_list.append(image)
    #     return ret_list
# ret_list = collect_data()
# for each in ret_list:
#     print(each.shape)
