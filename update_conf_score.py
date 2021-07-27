from detect import detect
from  API.update_conf import update_conf
def update_conf_score(path, collected_img,conf_thres):
    max_conf_list = []
    for img in collected_img:
        xyxy_list,label_list,conf_list = detect(img,conf_thres)
        if len(conf_list)>0:
            max_conf = max(conf_list)
            max_conf_list.append(max_conf)
    if len(max_conf_list)>0:
        update_conf(path, max(max_conf_list))
        return max(max_conf_list)
    else:
        return 0
