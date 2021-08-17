from detect import detect
from  API2.update_conf import update_conf
import os
import cv2
def update_conf_score(path,conf_thres):
    save_dir = 'frame_dir/secondary/'
    collected_img = []
    dirs = os.listdir( save_dir )

    for img in dirs:
       image = cv2.imread(save_dir+img)
       collected_img.append(image)

    max_conf_list = []
    for img in collected_img:
        xyxy_list,label_list,conf_list = detect(img,conf_thres)
        if len(conf_list)>0:
            max_conf = max(conf_list)
            ind = conf_list.index(max_conf)
            max_conf_list.append((max_conf,label_list[ind]))
    if len(max_conf_list)>0:
        max_conf_list.sort(reverse=True)
        final_conf = max_conf_list[0][0]
        label = max_conf_list[0][1]
        update_conf(path,final_conf,label)
        return final_conf, label
    else:
        return 0, 'None'
