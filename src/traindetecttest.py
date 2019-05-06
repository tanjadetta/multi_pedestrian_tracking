# issue #10


import cv2
import numpy as np
import util
import detection
import kamera
import Dataset

def makeManyTrainPics():
	srcPath = "../data/circles/train/"
	dstPath = "../data/circles/train/many/"
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
			
		
		
if __name__ == '__main__':
	makeManyTrainPics()