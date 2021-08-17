import requests
import shutil
import tempfile

def upload_img(path,id,x1,y1,x2,y2,label,confidence):
    # http://[Servername]:5000//set_Drone_Photo/[addrress]
    # $[drone id]
    # $[x1]$[y1]$[x2]$[y2]
    # $[label]
    # $[confidence_score]
    
    url = "http://127.0.0.1:5000/set_Drone_Photo/"+path+'${}'.format(id)\
        +'${}'.format(x1)\
        +'${}'.format(y1)\
        +'${}'.format(x2)\
        +'${}'.format(y2)\
        +"$'"+label\
        +"'${}".format(confidence)
    content = requests.get(url)
    return 'Uploaded Successfully'
