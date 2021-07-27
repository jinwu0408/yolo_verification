import requests
import shutil
import tempfile

def main():
    set_url = "http://127.0.0.1:5000/set_Drone_Photo/frame_9.jpg$2$1$1$2$2$'valve'$0.1"
    content = requests.get(set_url)
    print('seturl: ', content)

    get_url = 'http://127.0.0.1:5000/get_Drone_Photo/2'
    content1 = requests.get(get_url)
    print('geturl: ',len(content1.json()['data']))

    delete_url = 'http://127.0.0.1:5000/delete/2'
    content2 = requests.get(delete_url)
    print('deleteurl: ',content2)

    get_url = 'http://127.0.0.1:5000/get_Drone_Photo/2'
    content1 = requests.get(get_url)
    print('geturl: ',len(content1.json()['data']))


def upload_img(path,id,x1,y1,x2,y2,label,confidence):
    # http://[Servername]:5000//set_Drone_Photo/[addrress]
    # $[drone id]
    # $[x1]$[y1]$[x2]$[y2]
    # $[label]
    # $[confidence_score]
    url = "http://127.0.0.1:5000/set_Drone_Photo/"
        +path
        +'${}'.format(id)
        +'${}'.format(x1)
        +'${}'.format(y1)
        +'${}'.format(x2)
        +'${}'.format(y2)
        +'$'+label
        +'${}'.format(id)
    content = requests.get(set_url)
    return 'Uploaded Successfully'

def update_conf(path,confidence):
    # http://[Servername]:5000//set_Drone_Photo/
    # [addrress]
    # $[confidence_score]
    url = "http://[Servername]:5000//set_Drone_Photo/{}${}".format(path,confidence)
    content = requests.get(set_url)
    return 'Updated Successfully'

def clear_db(id):
    delete_url = 'http://127.0.0.1:5000/delete/{}'.format(id)
    content = requests.get(delete_url)
    return 'Cleared Successfully'

def get_size_with_id(id):
    get_url = 'http://127.0.0.1:5000/get_Drone_Photo/{}'.format(id)
    content1 = requests.get(get_url)
    return len(content1.json()['data'])
# if __name__=='__main__':
    # main()
