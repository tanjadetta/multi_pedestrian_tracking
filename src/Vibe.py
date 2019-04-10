import numpy as np
import Dataset
import cv2
import util
import random
class Vibe:
	
	N = 20		#number of samples
	T = 20			#Threshold 
	minN = 2		#minimum number of neighbors 
	updateFactor = 2
	samples = None	#samples which describe the background
	width = 0
	height = 0
	channels = 0
	
	def __init__(self, width, height, channels):
		self.height = height
		self.width  = width
		self.channels = channels
		#self.samples = np.zeros([self.N, height, width], dtype = np.uint8)
		self.samples = np.zeros([self.N, height, width, self.channels])
	
	#get a forground-mask for the given image 
	def getForegroundMask(self, im):
		N = self.N
		T = self.T
		anzPix = self.width * self.height
		A2 = np.zeros([N, anzPix])
		#iterate every sample
		for n in range(N):
			#reshape the image in a matrix:
			A = np.asmatrix(im.reshape(anzPix, self.channels)).T
			B = np.asmatrix(self.samples[n].reshape(anzPix, self.channels)).T
			#difference between im and sample:
			D = np.linalg.norm(A - B, axis = 0)
			A2[n] = D < T
		CountNearSamples = np.sum(A2, 0)
		Forground = CountNearSamples <= self.minN
		return Forground.reshape(self.height, self.width)
	
	def updateModel(self, im, forground_mask):
		#fgm = self.getForegroundMask(im)
		fgm = forground_mask
		rs = random.Random()
		neighbors = list()
		neighbors.append([1, 0])
		neighbors.append([0, 1])
		neighbors.append([-1, 0])
		neighbors.append([0, -1])
		neighbors.append([1, -1])
		neighbors.append([-1, 1])
		neighbors.append([1, 1])
		neighbors.append([-1, -1])
		
		print("Shape of fgm:", np.shape(fgm))
		for y in range(self.height):
			for x in range(self.width):
				if fgm[y, x] == False:
					#here is HG
					r = rs.randint(1, self.updateFactor)
					if r == 1:
						#update at [x,y]
						n = rs.randint(0, self.N - 1)
						self.samples[n, y, x] = im[y, x]
					r = rs.randint(1, self.updateFactor)
					if r == 1:
						#do update a neighbor of [x,y]:
						n = rs.randint(0, self.N - 1)
						neigh = neighbors[rs.randint(0, 7)]
						neighY = util.clamp(y + neigh[0], 0, self.height - 1)
						neighX = util.clamp(x + neigh[1], 0, self.width - 1)
						self.samples[n, neighY, neighX] = im[y, x]
						
	def updateModel2(self, im, forground_mask):
		#fgm = self.getForegroundMask(im)
		fgm = forground_mask.copy()
		fgm = ~fgm
		R1 = np.random.randint(0, self.updateFactor, np.shape(fgm))
		R2 = fgm * R1
		W  = np.where( R2 == 1 )
		ns = np.random.randint(0, self.N, np.size(W[0]))
		self.samples[(ns, W[0], W[1])] = im[W]
		
		#update neighbors
		R1 = np.random.randint(0, self.updateFactor, np.shape(fgm))
		R2 = fgm * R1
		W  = np.where( R2 == 1 )
		ns = np.random.randint(0, self.N, np.size(W[0]))
		moves = ( [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,-1], [1,-1], [-1,1] )
		ms = np.random.randint(0, 8, np.size(W[0]))
		
		WMY = []
		WMX = []
		for m,y,x in zip(ms,W[0],W[1]):
			WMY.append(util.clamp(y + moves[m][0], 0, self.height - 1))
			WMX.append(util.clamp(x + moves[m][1], 0, self.width - 1))
		self.samples[(ns, WMY, WMX)] = im[(W[0], W[1])]
		
		#for i in range(self.N):
		#	cv2.imshow("S" + str(i), self.samples[i].astype(np.uint8))
			
		
		
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
		#for i in range(self.N):
		#	cv2.imshow("S" + str(i), self.samples[i].astype(np.uint8))
	
	#def getBGpictureFromSequenz(ds, numberOfIms, step):
	#	imStack = np.zeros([numberOfIms, ds.])
	
if __name__ == '__main__':
	path = "c:/here_are_the_frames/"
	bg = Vibe(640,480,3)
	ds = Dataset.Dataset(path, "jpg")
	bgIm = cv2.imread("c:/here_are_the_frames/bg.jpg")
	#bgIm = ds.getFrame(0)
	cv2.imshow("BackgroundPicture", bgIm)
	bg.getModelFromBGPicture(bgIm)
	print (ds.info())
	i = 0
	while ds.getNextFrame():
		print("Frame: ", ds.aktuFrameIndex, " at ", ds.aktuFrame)
		im = ds.im
		cv2.imshow("Picture", im)
		F = bg.getForegroundMask(im)
		bg.updateModel2(im, F)
		cv2.imshow("Forground-Mask", 255*F.astype(np.uint8))
		taste = cv2.waitKey(20)
		if taste == 27:
			break
		cv2.imwrite(path + "bgs2_2/" + str(ds.getFilename()), 255*F.astype(np.uint8))
		i = i + 1