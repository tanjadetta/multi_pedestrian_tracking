import glob
import numpy as np
import cv2

class Dataset:
	path = ""
	startFrame = 0
	endFrame   = 0
	aktuFrameIndex = 0
	frameList = None
	
	def __init__(self, path, fileExt):
		self.path = path
		self.aktuFrameIndex = -1
		self.fileExt = fileExt
		self.frameList = glob.glob(path + "/*." + fileExt)
		self.aktuFrame = ""
		self.frameCount = 0
		self.im = None
		self.frameCount = len(self.frameList)
		
	def info(self):
		return "Dataset with \n" \
		"path = "       + str(self.path) + "\n"  \
		"file Ext: = " + str(self.fileExt) + "\n" \
		"Count " + str(self.frameCount) + " Frames."
		
	def getNextFrame(self):
		if self.aktuFrameIndex < self.frameCount - 1:
			self.aktuFrameIndex = self.aktuFrameIndex + 1
			self.aktuFrame = self.frameList[self.aktuFrameIndex]
			self.im = cv2.imread(self.aktuFrame)
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
		