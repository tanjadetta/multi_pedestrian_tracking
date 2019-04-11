import util
import Dataset
import numpy as np
import cv2

class Detection:
	img = None
	mag = None
	ang = None
	def __init__(self, im):
		self.img = im.copy()
		self.mag, self.ang = util.magAndAngle(self.img)
		
	def makeHog(self):
		#calculate the hog of the patch
		h, w = np.shape(self.mag)
		bins = np.zeros(9)
		for y in range(h):
			for x in range(w):
				m = self.mag[y, x]
				if m > 0.0001:
					a = self.ang[y, x]
					#bins are at angle 	0 20 40 60 80 100, 120, 140, 160
					#					0  1  2  3  4   5    6    7    8
					b0 = int(a / 20)
					b1 = (b0 + 1) % 9
					r = a % 20
					p = r/20.0
					#example:	if a = 80 and m = 2 then 2 goes into bin 80
					#			if a = 10 and m = 4 then 2 goes into bin  0 and 2 into bin 20
					#			if a = 25 then 75% = 1 - 5/20 of b goes into b0 = 1 and 25% into bin 2 
					m0 = (1.0 - p) * m
					m1 = m - m0
					bins[b0] += m0
					bins[b1] += m1
			print(y)
		print("Summe der Magnituden: ", np.sum(self.mag))
		return bins
				
if __name__ == "__main__":
	print("Start")
	ds = Dataset.Dataset("c:/here_are_the_frames", "jpg")
	im = ds.getFrame(0)
	im = cv2.imread("c:/here_are_the_frames/test/002.jpg")
	dets = Detection(im)
	#dets.ang = np.array([[80,15,0,175],[20,30,100,105]])
	#dets.mag = np.array([[4,8,12,16]  ,[0,4,8,12]])
	bins = dets.makeHog()
	print(np.round(bins))
	