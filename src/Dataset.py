import glob
import numpy as np
import cv2
import os

class Dataset:
	path = ""
	aktuFrameIndex = 0
	frameList = None
	imShape = None
	frameCount = 0
	
	def __init__(self, path, fileExt):
		self.path = path
		self.aktuFrameIndex = -1
		self.fileExt = fileExt
		self.frameList = sorted(glob.glob(path + "/*." + fileExt))
		self.aktuFrame = ""
		self.frameCount = 0
		self.im = None
		self.frameCount = len(self.frameList)
		tmpIm = self.getFrame(0)
		self.imShape = np.shape(tmpIm)
	
	def getFrame(self, index):
		if index < self.frameCount and index >= 0:
			d = self.frameList[index]
			return cv2.imread(d)
		else:
			return None
		
	def getFilename(self, i = None):
		if i is None:
			return os.path.basename(self.aktuFrame)
		else:
			return os.path.basename(self.frameList[i])
	
	
	def info(self):
		return "Dataset with \n" \
		"path = "       + str(self.path) + "\n"  \
		"file Ext: = " + str(self.fileExt) + "\n" \
		"Count " + str(self.frameCount) + " Frames."
		
	def getNextFrame(self):
		self.aktuFrameIndex = self.aktuFrameIndex + 1
		self.im = self.getFrame(self.aktuFrameIndex)
		if not (self.im is None):
			self.aktuFrame = self.frameList[self.aktuFrameIndex]
			return True
		else:
			return False

if __name__ == '__main__':
	ds = Dataset("c:/here_are_the_frames/", "jpg")
	print (ds.info())
	while ds.getNextFrame():
		print("Frame: ", ds.aktuFrameIndex, " at ", ds.aktuFrame)
		cv2.imshow("Hallo", ds.im)
		cv2.waitKey()
		