from fr_utils import img_to_encoding
import os
def create_database(path,FRmodel):
	database = {}
	identity = os.path.splitext(os.path.basename(path))[0]
	database[identity] = img_to_encoding(path,FRmodel)
	print(database)
	return database
