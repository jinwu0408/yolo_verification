import requests
import shutil
import tempfile

def update_conf(path,confidence,label):
    # http://[Servername]:5000//set_Drone_Photo/
    # [addrress]
    # $[confidence_score]
    url = "http://127.0.0.1:5000/update_confidence_score/{}${}${}".format(path,confidence,label)
    content = requests.get(url)
    return 'Updated Successfully'
