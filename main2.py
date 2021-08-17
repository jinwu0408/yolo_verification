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
from API2.clear_db import clear_db
from API2.upload_img import upload_img
from API2.update_conf import update_conf
from API2.get_size_with_id import get_size_with_id

#source ~/code/parrot-groundsdk/./products/olympe/linux/env/shell
if __name__ == "__main__":
    print('Starting....\n')

    conf_cutoff = 0.5 #for senting the drone
    conf_thres=0.2 #for yolo
    webcam_id = 0
    primary_drone_id = 1 #primary_drone_id for the databasse
    secondary_drone_id = 2 #secondary_drone_id for the databasse
    clear_db(primary_drone_id) #clear the database
    clear_db(secondary_drone_id)
    print('Starting the Webcam')
    cap = cv2.VideoCapture(webcam_id)
    # cont = True
    while(True):
        ret, frame = cap.read()
        #Run yolo detection
        xyxy_list,label_list,conf_list = detect(frame,conf_thres)
        num_detect = len(label_list)
        print('detected: {} with {}'.format(label_list,conf_list))
        ###############Check if the confidence score is less than conf_cutoff####################
        for i in range(num_detect):
            xyxy = xyxy_list[i]#ndarray
            label = label_list[i]#str
            conf = conf_list[i]#float
            ###############################Need to sent a drone############################3
            if conf<conf_cutoff:
                print('Current database size: {}'.format(get_size_with_id(primary_drone_id)))
                save_path = 'frame_dir|primary|tmp_frame.jpg'
                cv2.imwrite(save_path.replace('|','/'), frame)
                upload_img(save_path,
                            primary_drone_id,
                            xyxy[0],
                            xyxy[1],
                            xyxy[2],
                            xyxy[3],
                            label,
                            conf)
                print("Database size after upload: {}".format(get_size_with_id(primary_drone_id)))
                #sent_drone
                print("Sending the Drone")
                # time.sleep(1)
                # collect_data
                # print('Collecting Data')
                # cv2.destroyAllWindows()
                collect_data(secondary_drone_id)
                #update_conf_score
                new_conf,new_label = update_conf_score(save_path,conf_thres)
                if new_conf != 0 and new_conf>conf:
                    plot_one_box(xyxy, frame, label=new_label+' '+ "{:.2f}".format(new_conf), color=colors(0, True))
                    print('Old Conf:{}  New Conf:{}'.format(conf,new_conf))
                    print('New Confidence Score Been Updated')
                else:
                    # plot_one_box(xyxy, frame, label=label+' '+ "{:.2f}".format(conf), color=colors(0, True))
                    print('False Detection')

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
