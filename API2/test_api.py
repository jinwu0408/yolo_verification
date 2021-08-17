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
if __name__=='__main__':
    main()
