import requests
import shutil
import tempfile
def clear_db(id):
    '''
    Clear all data with the drone id in the database.

    Parameters:
        id (int): The drone id

    Returns:
        (String): "Cleared Successfully"
    '''

    print('Clearing the database.')
    delete_url = 'http://127.0.0.1:5000/delete/{}'.format(id)
    content = requests.get(delete_url)
    return 'Cleared Successfully'
