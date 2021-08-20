import requests
import shutil
import tempfile

def update_conf(path,confidence,label):
    '''
    Update the confidence score and label of the image associates with the path
    in the database.

    Parameters:
        path (String): The path of the image
        confidence(float): The confidence cutoff for yolo to determine
        detections.
        label(String): The label for the detections

    Returns:
        (String): 'Updated Successfully'
    '''
    # http://[Servername]:5000//set_Drone_Photo/
    # [addrress]
    # $[confidence_score]

    # print('Updating the confidence score')
    url = "http://127.0.0.1:5000/update_confidence_score/{}${}${}".format(path,confidence,label)
    content = requests.get(url)
    return 'Updated Successfully'
