import requests
import shutil
import tempfile
def clear_db(id):
    delete_url = 'http://127.0.0.1:5000/delete/{}'.format(id)
    content = requests.get(delete_url)
    return 'Cleared Successfully'
