import time
import datetime
import requests

url = 'https://notify-api.line.me/api/notify'
token = 'ACCagTTxrgFqU7Zkk8ZoU3UjQnqpIQLGDoe4eL3fOLn'
headers = {'Authorization':'Bearer '+token}

count_s = time.time()
time_stamp = datetime.datetime.fromtimestamp(count_s).strftime('%Y-%m-%d %H:%M:%S')
requests.post(url, headers=headers ,data = {'message':'Detected Drone_Scout at : '+ time_stamp} ,files = {'imageFile':open("detected_img/tmp.jpg",'rb')})