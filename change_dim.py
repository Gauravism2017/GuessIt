from PIL import Image
import os, sys

path = "./UTKFace/"
f = "./data128/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            #f, e = os.path.splitext(path+item)
            imResize = im.resize((128,128), Image.ANTIALIAS)
            imResize.save(f + item, 'JPEG', quality=90)

resize()