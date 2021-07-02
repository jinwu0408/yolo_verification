import argparse
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from detect import detect
from calculate_target_pos import calculate_target_pos
from sent_drone import sent_drone
from collect_data import collect_data
from update_conf_score import update_conf_score


if __name__ == "__main__":
    print('Starting....\n')

    conf_cutoff = 0.5
    webcam_id = 0

    print('Starting the Webcam')
    cap = cv2.VideoCapture(webcam_id)
    cont = True
    while(True):
        ret, frame = cap.read()

        xyxy_list,label_list,conf_list = detect(frame)
        num_detect = len(label_list)
        for i in range(num_detect):
            xyxy = xyxy_list[i]
            label = label_list[i]
            conf = conf_list[i]
            if conf<conf_cutoff:
                # frame_cp = frame.copy()
                # plot_one_box(xyxy, frame_cp, label=label+' '+ "{:.2f}".format(conf) + '?????', color=colors(0, True))
                # print
                # cv2.imshow('frame2',frame_cp)

                #sent_drone
                print("Sending the Drone")
                # time.sleep(1)
                #collect_data
                print('Collecting Data')
                # cv2.destroyAllWindows()
                collected_img = collect_data()
                #update_conf_score
                new_conf = update_conf_score(collected_img)
                if new_conf != 0:
                    plot_one_box(xyxy, frame, label=label+' '+ "{:.2f}".format(new_conf), color=colors(0, True))
                    print('New Confidence Score Been Updated')
                else:
                    plot_one_box(xyxy, frame, label=label+' '+ "{:.2f}".format(conf), color=colors(0, True))
                    print('Not be able to update the confidence')
                # key_input = input('Press Enter to continue, quit to QUIT')
                # print(key_input)
                # if key_input == 'quit':
                #     cont = False
                #     break
            else:
                plot_one_box(xyxy, frame, label=label+' '+ "{:.2f}".format(conf), color=colors(0, True))

        cv2.imshow('frame',frame)
        # if cont == False:
        #     print('Closing the Webcam')
        #     cv2.destroyAllWindows()
        #     break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Closing the Webcam')
            cv2.destroyAllWindows()
            break

    print('FINISH')
