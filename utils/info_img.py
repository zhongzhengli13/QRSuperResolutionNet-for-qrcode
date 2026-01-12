import requests
from urllib.parse import quote

r = requests.get("https://api.2dcode.biz/v1/read-qr-code?file_url=" + quote("https://gstatic.clewm.net/caoliao-resource/250408/80bc7c_bd33d499.png"))
print(r.json()["data"]["contents"][0])