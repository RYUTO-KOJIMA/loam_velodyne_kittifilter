
import time
import codecs
import os

SAVE_DATA_PATH = "/home/kojima/saved_data/" + str( int(time.time()) )
SAVE_FILE_NAME = SAVE_DATA_PATH + "/" +  "drqn_learn_data.txt"
CREATE_DATA_FILE = True

def fileprint(*s):

    if CREATE_DATA_FILE:
        os.makedirs(SAVE_DATA_PATH,exist_ok=True)    
        with codecs.open(SAVE_FILE_NAME, 'a', 'utf-8') as f:
            print (*s,file=f)

