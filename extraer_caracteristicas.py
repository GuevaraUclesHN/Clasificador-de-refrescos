import cv2 
import sys
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
import csv
from collections import defaultdict
from multiprocessing import Pool
import os

def Cutter(fir):
        print(fir)
        img = cv2.imread(fir)
        mask =	np.zeros(img.shape[:2],np.uint8)

        bgdModel =  np.zeros((1,65),np.float64)
        fgdModel =  np.zeros((1,65),np.float64)


        rect =	(130,0,300,900)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

        mask2  =  np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img  =	img*mask2[:,:,np.newaxis]
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #   plt.imshow(img)
    #   plt.show()
        histo(img,fir)

def histo(img,fir):
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    Rhistogram = exposure.histogram(R)
    Ghistogram = exposure.histogram(G)
    Bhistogram = exposure.histogram(B)

    with open(sys.argv[2],"a") as f:
        for histogram in (Rhistogram, Ghistogram, Bhistogram):
            values, bins = histogram
            for v, b in zip(values, bins):
                f.write("%d:%d," % (b, v))
        f.write(fir)
        f.write("\n")


def pooler():
    dirPath = "./" + sys.argv[1]
    files = os.listdir(dirPath)
    for File in files:
        imgPath = os.path.join(dirPath, File)
        Cutter(imgPath)
        
pooler()

