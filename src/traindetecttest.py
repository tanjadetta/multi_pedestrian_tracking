# issue #10


import cv2
import numpy as np
import util
import detection
import kamera
import Dataset
import os
import glob
from numpy.random.mtrand import randint
from util import plotHelper

def makeManyTrainPics(srcPath, dstPath):
	pos = 0
	neg = 0
	dataS = Dataset.Dataset(srcPath, "jpg")
	while dataS.getNextFrame():
		util.plotHelper(dataS.im, "bla")
		#if "pos" in dataS.getFilename():
		if "pos" in dataS.getFilename():
			for angle in range(0,360,20):
				im2 = util.rotatePicture(dataS.im, angle, True, (255,255,255))
				util.plotHelper(im2, "rotated")
				cv2.imwrite(dstPath + "pos_" + "{:>04d}".format(pos) + ".jpg", im2)
				pos = pos + 1
		elif "neg" in dataS.getFilename():
			for angle in range(0,360,20):
				im2 = util.rotatePicture(dataS.im, angle, True, (255,255,255))
				util.plotHelper(im2, "rotated")
				cv2.imwrite(dstPath + "neg_" + "{:>04d}".format(neg) + ".jpg", im2)
				neg = neg + 1
		else:
			raise RuntimeError("File " + str(dataS.getFilename()) +  " is neither pos nor neg !?")
			


def projectTestsOnFrame():
	#srcPath = "../data/circles/test/"
	#dstPath = "../data/circles/test/many/"
	#for d in glob.glob(dstPath + "*.jpg"):
	#	print ("Loesche: " + d) 
	#	os.remove(d)
	#makeManyTrainPics(srcPath, dstPath)
	cam = kamera.kamera("../data/calibration/iscam2.cali", "")
	dstPath = "C:/here_are_the_frames/00000002.jpg"
	dstIm = cv2.imread(dstPath)
	testCount = 5
	srcPath = "../data/circles/test/many/"
	ds = Dataset.Dataset(srcPath, "jpg")
	r = np.random.randint(0, ds.frameCount, testCount)
	dstH, dstW, _ = np.shape(dstIm) 
	for i in range(testCount):
		srcIm = ds.getFrame(r[i])
		util.plotHelper(srcIm, "bla_" + str(i), False)
		fpUV = np.matrix([(dstW-100) * np.random.rand() + 50, 
					      (dstH-100) * np.random.rand() + 50]).T
		fp = cam.unproj(fpUV)
		cam.projectPicture(fp, 1, 1, srcIm, dstIm)
	util.plotHelper(dstIm)	
		
		
if __name__ == '__main__':
	#makeManyTrainPics()
	projectTestsOnFrame()