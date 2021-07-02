from detect import detect
def update_conf_score(collected_img):
    max_conf_list = []
    for img in collected_img:
        xyxy_list,label_list,conf_list = detect(img)
        if len(conf_list)>0:
            max_conf = max(conf_list)
            max_conf_list.append(max_conf)
    if len(max_conf_list)>0:
        return max(max_conf_list)
    else:
        return 0 
