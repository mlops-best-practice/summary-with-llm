import os
import json
import base64
from utils.common.os_util import load_file_txt


def load_gcloud_credential():
    path = os.getenv("GCLOUD_BASE64_CREDENTIAL_PATH", None)
    base64_credential = os.getenv("GCLOUD_BASE64_CREDENTIAL", None)
    if path:
        content = load_file_txt(path)
        base64_bytes = base64.b64decode(content)
    elif base64_credential:
        base64_bytes = base64.b64decode(base64_credential)
    
    else:
        raise Exception("Not exited credentials google cloud")

    json_data = json.loads(base64_bytes.decode('utf-8'))
    return json_data


def load_mysql_credential():
    path = os.getenv("MYSQL_BASE64_CREDENTIAL_PATH", None)
    base64_credential = os.getenv("MYSQL_BASE64_CREDENTIAL", None)
    
    if path:
        content = load_file_txt(path)
        base64_bytes = base64.b64decode(content)
    elif base64_credential:
        base64_bytes = base64.b64decode(base64_credential)
        
    else:
        raise Exception("MYSQL CREDENTIAL BASE 64 NOT DEFINE")
    
    json_data = json.loads(base64_bytes.decode('utf-8'))
    return json_data