# import olympe
# import time
# from olympe.messages.ardrone3.Piloting import TakeOff, Landing
#
# DRONE_IP = "192.168.42.1"
#
#
# def main():
#     drone = olympe.Drone(DRONE_IP)
#     drone.connect()
#     assert drone(TakeOff()).wait().success()
#     time.sleep(2)
#
#     assert drone(Landing()).wait().success()
#     drone.disconnect()
#
#
# if __name__ == "__main__":
#     main()
#

# -*- coding: UTF-8 -*-

import olympe
import time
from olympe.messages.ardrone3.Piloting import TakeOff, moveBy, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

DRONE_IP = "192.168.42.1"

if __name__ == "__main__":
    drone = olympe.Drone(DRONE_IP)
    drone.connect()


    assert drone(Landing()).wait().success()
    drone.disconnect()
