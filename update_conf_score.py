from detect import detect
from  API2.update_conf import update_conf
import os
import cv2
def update_conf_score(path,conf_thres):
    '''
    Update the confidence score of the image associates with the path
    in the database using the images saved in the '~/frame_dir/secondary/'.

    Parameters:
        path (String): The path of the image
        conf_thres(float): The confidence cutoff for yolo to determine
        detections.

    Returns:
        final_conf (float): The confidence score that is putted into the
        database. '0' if no detection.
        label (String): The final label. 'None' is no detection
    '''
    save_dir = 'frame_dir/secondary/'
    collected_img = []
    dirs = os.listdir( save_dir )


    print('Updating the confidence')
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
        print('Updated to {} with {} confidence'.format(label,final_conf))
        return final_conf, label
    else:
        update_conf(path,0,'None')
        return 0, 'None'
