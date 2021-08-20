import requests
import shutil
import tempfile

def upload_img(path,id,x1,y1,x2,y2,label,confidence):
    '''
    Upload the image into the database using the API.

    Parameters:
        path (String): The path of the image
        id(int): The drone id.
        x1(float): The center location.
        x2(float): The center location.
        y1(float): The center location.
        y2(float): The center location.
        label(String): The label for the detections.
        confidence(float): The confidence for the detection.

    Returns:
        (String): 'Uploaded Successfully'
    '''
    # http://[Servername]:5000//set_Drone_Photo/[addrress]
    # $[drone id]
    # $[x1]$[y1]$[x2]$[y2]
    # $[label]
    # $[confidence_score]
    print('Uploading the detection to the database')
    url = "http://127.0.0.1:5000/set_Drone_Photo/"+path+'${}'.format(id)\
        +'${}'.format(x1)\
        +'${}'.format(y1)\
        +'${}'.format(x2)\
        +'${}'.format(y2)\
        +"$'"+label\
        +"'${}".format(confidence)
    content = requests.get(url)
    return 'Uploaded Successfully'
