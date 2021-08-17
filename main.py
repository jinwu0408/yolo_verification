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
    #connect to the drone
    object = 'Good'
    source='0'
    drone_weights = 'valve_2.pt'
    object_weights = 'valve'
    imgsz=640
    conf_thres=0.25
    iou_thres=0.45
    conf_cutoff = 0.5
##############################initialize Model##########################
    print('Initializing Yolo Model...\n')
    device ='cuba' if torch.cuda.is_available() else 'cpu'
    half = device != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
########################################################################
    quit=False

        ############################FOR EACH FRAME#########################
    for path, img, im0s, vid_cap in dataset:
        if not quit:
            print('reading webcam')
            ####################################Make detections#################
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            ############################FOR EACH DETECTION#######################
            for i, det in enumerate(pred):
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # img.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    print('detected')
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # center = np.array((int((det[0][0] + det[0][2])/2), int((det[0][1] + det[0][3])/2)))

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        print('Confidence is {}'.format(conf))

                        c = int(cls)  # integer class
                        label = None if False else (names[c] if False else f'{names[c]} {conf:.2f}')
                        print('Class is {}\n'.format(label[0:-5]))
                        if label[0:-5] == object and conf > conf_cutoff:
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True))
                        if label[0:-5] == object and conf <= conf_cutoff:
                            plot_one_box(xyxy, im0, label=label+'?', color=colors(c, True))
                            cv2.imshow(str(p), im0)
                            xyxy1 = np.array(xyxy)
                            center = np.array((int((xyxy1[0] + xyxy1[2])/2), int((xyxy1[1] + xyxy1[3])/2)))
                            pos = calculate_target_pos(center)#this funciton should return a tuple of (x1,y1,angle)
                            success = sent_drone(pos)#should use PID control to send the drone to the target position, return boolean
                            if success:
                                collect_data()
                                new_conf_score = update_conf_score()#input: image data, output:new score
                            else:
                                #Land the drone
                                pass

                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == 27:#ord('q'):  # q to quit
                    print('Shutting Down the camera')
                    cv2.destroyAllWindows()
                    quit = True
        else:
            break
    print('Ended')


    #     xyxy,conf,label = detect()
    #     #check if object's confidence is above threshold
    #     if label == 'object' and conf>conf_thres:
    #         #put label on the frame
    #         pass
    #     elif label == 'object' and conf<=conf_thres:
    #         #put label with question mark
    #         #sent the drone to check
    #         pos = calculate_target_pos(cur_drone_pos)#this funciton should return a tuple of (x1,y1,angle)
    #         success = sent_drone(pos)#should use PID control to send the drone to the target position, return boolean
    #         if success:
    #             collect_data()
    #             new_conf_score = update_conf_score()#input: image data, output:new score
    #             #put the new label on the web_frame
    #     #ask to continue
    #     if input('press q to quit')=='q':
    #         quit = True
    # print('Program Ended')
