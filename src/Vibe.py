import numpy as np
import Dataset
import cv2
class Vibe:
	
	N = 20			#number of samples
	T = 20			#Threshold 
	minN = 2		#minimum number of neighbors 
	updateFactor            = 16
	samples = None	#samples which describe the background
	width = 0
	height = 0
	channels = 1
	
	def __init__(self, width, height):
		self.height = height
		self.width  = width
		self.samples = np.zeros([self.N, height, width])
	
	#get a forground-mask for the given image im
	def getForegroundMaskForGrayIm(self, grayIm):
		N = self.N
		T = self.T
		A = np.zeros([N, self.height, self.width], dtype=np.uint8)
		for n in range(N):							#iterate every sample
			D = np.abs(grayIm - self.samples[n]).astype(np.uint8)	#difference between grayIm and sample i
			A[n] = (D > T).astype(np.uint8)
		CountNearSamples = np.sum(A, 0).astype(np.uint8)
		Forground = CountNearSamples >= self.minN
		return Forground
	
	
	
	def getModelFromBGPicture(self, im):
		for n in range(self.N):
			self.samples[n] = im.copy()
	
if __name__ == '__main__':
	bg = Vibe(640,480)
	bgIm = cv2.imread("C:/here_are_the_frames/bg.png")
	bgIm = bgIm[:,:,0]								#first test with one channel
	cv2.imshow("BackgroundPicture", bgIm)
	cv2.waitKey()
	bg.getModelFromBGPicture(bgIm)
	ds = Dataset.Dataset("c:/here_are_the_frames/", "jpg")
	print (ds.info())
	while ds.getNextFrame():
		print("Frame: ", ds.aktuFrameIndex, " at ", ds.aktuFrame)
		im = ds.im[:,:,0]
		cv2.imshow("Picture", im)
		cv2.waitKey()
		F = bg.getForegroundMaskForGrayIm(im)
		cv2.imshow("Forground-Mask", 255*F.astype(np.uint8))
		cv2.waitKey()