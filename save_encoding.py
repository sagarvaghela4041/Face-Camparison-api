# import numpy as np

# # file = open('me.txt','r+')
# # arr = str(np.array((1, 2, 3, 4, 5)))
# # print(arr)
# # #file.write(arr)
# # a = np.fromstring(arr[1:-1], dtype=np.int, sep=' ')
# # print(a)


# # file = open('images/me.txt','r+')
# # me = file.readlines()


# #me = np.fromstring(me[1:-1], dtype=np.int, sep=' ')
# me = ""
# rdj = ""
# with open('images/me.txt') as f:
#     me = (f.read().rstrip())
# me = np.array(me)
# with open('images/rdj.txt') as f:
#     rdj = (f.read().rstrip())
# rdj = np.array(rdj)
# print(me,rdj)
# dist = np.linalg.norm((me - rdj))
# print(dist)



import os
def create_database(path):
	database = {}
	identity = os.path.splitext(os.path.basename(path))[0]
	#database[identity] = img_to_encoding(file_path,FRmodel)
	print(path,identity)
	return database

create_database('images/me.jpg')