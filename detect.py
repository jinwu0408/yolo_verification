import argparse
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


# source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
weights = 'can.pt'
imgsz=640
# conf_thres=0.7
iou_thres=0.45
# save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
save_img=False
save_txt = False
webcam = False

# Directories
# save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
# (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Initialize
# set_logging()

device ='cuba' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
if half:
    model.half()  # to FP16

# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()


# Run inference
if device != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
t0 = time.time()
center_list=[]
def detect(im0,conf_thres):
    # Padded resize
    img = letterbox(im0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    # print(img.shape)
    # img = np.reshape(img,(imgsz,imgsz,3))
    # print(img.shape)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # print(img.shape)
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    t2 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    label_list = []
    conf_list = []
    xyxy_list = []

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        # if webcam:  # batch_size >= 1
        #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        # else:
        #     p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
        #
        # p = Path(p)  # to Path
        # save_path = str(save_dir / p.name)  # img.jpg
        # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        # s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # center = np.array((int((det[0][0] + det[0][2])/2), int((det[0][1] + det[0][3])/2)))

            # Print results
            # for c in det[:, -1].unique():
            #     n = (det[:, -1] == c).sum()  # detections per class
            #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                #     with open(txt_path + '.txt', 'a') as f:
                #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # if save_img  or view_img:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if False else (names[c] if False else f'{names[c]}')

                # print(xyxy.shape)
                # center = np.array((int((xyxy[0][0] + xyxy[0][2])/2), int((xyxy[0][1] + xyxy[0][3])/2)))
                # center_list.append(center
                plot_one_box(xyxy, im0, label=label, color=colors(c, True))
                xyxy1 = np.array(xyxy)
                # print(xyxy.shape)
                xyxy_list.append(xyxy1)
                label_list.append(label)
                conf_list.append(conf.item())
                center = np.array((int((xyxy1[0] + xyxy1[2])/2), int((xyxy1[1] + xyxy1[3])/2)))
                center_list.append(center)

    return xyxy_list,label_list,conf_list


# frame = cv2.imread('./all_data/night/frame_0.jpg')
# print(frame.shape)
# cv2.imshow('frame',frame)
# print(detect(frame))
