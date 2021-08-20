
import requests
import shutil
import tempfile


def get_size_with_id(id):
    '''
    Get the size of data stored for drone id.

    Parameters:
        id (int): The drone id.

    Returns:
        size(int): The size of data stored for drone id.
    '''
    get_url = 'http://127.0.0.1:5000/get_Drone_Photo/{}'.format(id)
    content1 = requests.get(get_url)
    # print(content1.json())
    return len(content1.json()['data'])
