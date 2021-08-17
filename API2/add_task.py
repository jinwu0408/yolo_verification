import requests
import shutil
import tempfile

def add_task(drone_id=0,taskname="Null",task="Null"):
    # http://[Servername]:5000//set_Drone_Photo/
    # [addrress]
    # $[confidence_score]
    url = "http://127.0.0.1:5000/add_task/{}${}${}".format(drone_id,taskname,task)
    content = requests.get(url)
    return 'Updated Successfully'
