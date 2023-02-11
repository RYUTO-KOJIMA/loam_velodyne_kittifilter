
import time
import codecs

SAVE_DATA_PATH = "/home/kojima/saved_data"
SAVE_FILE_NAME = SAVE_DATA_PATH + "/" +  "drqn_learn_data_" + str( int(time.time()) ) + ".txt"

def fileprint(*s):
    with codecs.open(SAVE_FILE_NAME, 'a', 'utf-8') as f:
        print (*s,file=f)