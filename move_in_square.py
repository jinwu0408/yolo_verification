import os
import olympe
import time
import shlex
import subprocess
from olympe.messages.ardrone3.Piloting import TakeOff, moveBy, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.camera import set_alignment_offsets

DRONE_IP = "192.168.42.1"

if __name__ == "__main__":
    drone = olympe.Drone(DRONE_IP)
    drone.connect()
    # assert drone(
    #     TakeOff()
    #     >> FlyingStateChanged(state="hovering", _timeout=5)
    # ).wait().success()
    # time.sleep(3)

    # drone.set_streaming_output_files(
    #             h264_data_file=os.path.join('out', 'h264_data.264'),
    #             h264_meta_file=os.path.join('out', 'h264_metadata.json'),
    #             # Here, we don't record the (huge) raw YUV video stream
    #             # raw_data_file=os.path.join(self.tempd,'raw_data.bin'),
    #             # raw_meta_file=os.path.join(self.tempd,'raw_metadata.json'),
    #         )
    drone.start_video_streaming()
    # print(drone(set_alignment_offsets(cam_id=0,yaw=30,pitch=30,roll=30)).wait().success())
    time.sleep(5)



    #
    drone.stop_video_streaming()


    # assert drone(Landing()).wait().success()
    drone.disconnect()

    # h264_filepath = os.path.join('out', 'h264_data.264')
    # mp4_filepath = os.path.join('out', 'h264_metadata.mp4')
    # subprocess.run(
    #     shlex.split('ffmpeg -i {} -c:v copy -y {}'.format(
    #         h264_filepath, mp4_filepath)),
    #     check=True
    # )
