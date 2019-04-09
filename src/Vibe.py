import numpy as np
import Dataset
import cv2
import util
class Vibe:
	
	N = 20			#number of samples
	T = 30			#Threshold 
	minN = 2		#minimum number of neighbors 
	updateFactor            = 16
	samples = None	#samples which describe the background
	width = 0
	height = 0
	channels = 3
	
	def __init__(self, width, height):
		self.height = height
		self.width  = width
		#self.samples = np.zeros([self.N, height, width], dtype = np.uint8)
		self.samples = np.zeros([self.N, height, width, self.channels])
	
	#get a forground-mask for the given image 
	def getForegroundMaskForGrayIm(self, im):
		N = self.N
		T = self.T
		anzPix = self.width * self.height
		A2 = np.zeros([N, anzPix])
		for n in range(N):											#iterate every sample
			A = np.asmatrix(im.reshape(anzPix, self.channels)).T	#reshape the image in a matrix
			B = np.asmatrix(self.samples[n].reshape(anzPix, self.channels)).T
			D = np.linalg.norm(A - B, axis = 0)						#difference between im and sample 
			A2[n] = D < T
		CountNearSamples = np.sum(A2, 0)
		Forground = CountNearSamples <= self.minN
		return Forground.reshape(self.height, self.width)
	
	
	
	def getModelFromBGPicture(self, im):
		M = list()
		M.append(util.movePicture(im, 0, 0))
		M.append(util.movePicture(im, 1, 0))
		M.append(util.movePicture(im, 0, 1))
		M.append(util.movePicture(im, 1, 1))
		M.append(util.movePicture(im, -1, 0))
		M.append(util.movePicture(im, 0, -1))
		M.append(util.movePicture(im, -1, -1))
		M.append(util.movePicture(im, -1, +1))
		M.append(util.movePicture(im, +1, -1))
		
		r = np.random.randint(0, 9, self.N)
		
		for n in range(self.minN):
			#self.samples[n] = im
			print(np.shape(M[0]))
			self.samples[n] = M[0]
		
		for n in range(self.minN, self.N):
			self.samples[n] = M[r[n]]
		
		for n in range(self.N):
			cv2.putText(self.samples[n], str(n), (0,20), 1, 2, 255)
			cv2.imshow("Picture", self.samples[n].astype(np.uint8))
			cv2.waitKey()
			
if __name__ == '__main__':
	bg = Vibe(640,480)
	bgIm = cv2.imread("C:/here_are_the_frames/bg.png")
	#bgIm = bgIm[:,:,0]								#first test with one channel
	cv2.imshow("BackgroundPicture", bgIm)
	cv2.waitKey()
	bg.getModelFromBGPicture(bgIm)
	ds = Dataset.Dataset("c:/here_are_the_frames/", "jpg")
	print (ds.info())
	while ds.getNextFrame():
		print("Frame: ", ds.aktuFrameIndex, " at ", ds.aktuFrame)
		im = ds.im
		cv2.imshow("Picture", im)
		cv2.waitKey()
		F = bg.getForegroundMaskForGrayIm(im)
		cv2.imshow("Forground-Mask", 255*F.astype(np.uint8))
		cv2.waitKey()