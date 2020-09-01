from face_comp import verify,FRmodel
from my_utils import create_database
from bottle import route, run, request
import cv2
import urllib.request
import numpy as np


# def img_downloader(url):
# 	with urllib.request.urlopen(url) as url:
# 		arr = np.asarray(bytearray(url.read()), dtype=np.uint8)
# 		img = cv2.imdecode(arr, -1) # 'Load it as it is'
# 		cv2.imwrite("images/download"+url+".jpg",img)
@route('/face_comp')
def index():
	# img_downloader(request.query.img)
	img = request.query.img
	name = request.query.name
	database = create_database('images/'+name+'.jpg',FRmodel)
	verify(img, name, database, FRmodel)
	return "Success !"

run(host='localhost', port=8080, debug = True, reloader = True)