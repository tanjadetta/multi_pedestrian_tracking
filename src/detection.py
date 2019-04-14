import util
import Dataset
import numpy as np
import cv2
from math import cos, sin, pi

class Detection:
	img = None
	mag = None
	ang = None
	def __init__(self, im):
		self.img = im.copy()
		self.mag, self.ang = util.magAndAngle(self.img)
		
		
	def makeHogs(self, startY, height, startX, width, numOfCellsY, numOfCellsX):
		deltaY = height / numOfCellsY				#height of one cell
		deltaX = width / numOfCellsX				#width of one cell
		hogs = np.zeros([numOfCellsY, numOfCellsX, 9])
		for y in range(numOfCellsY):
			sY = startY + y * deltaY 			
			eY = round(sY + deltaY)
			sY = round(sY)			
			for x in range(numOfCellsX):
				sX = startX + x * deltaX 			
				eX = round(sX + deltaX)
				sX = round(sX)
				hogs[y,x] = self.makeHog(sY, eY, sX, eX)
		return hogs
	
	def makeHog(self, sY, eY, sX, eX):
		#calculate the hog of the patch
		bins = np.zeros(9)
		for y in range(sY, eY):
			for x in range(sX, eX):
				m = self.mag[y, x]
				if m > 0.0001:
					a = self.ang[y, x]
					#bins are at angle 	0 20 40 60 80 100, 120, 140, 160
					#					0  1  2  3  4   5    6    7    8
					b0 = int(a / 20)
					b1 = (b0 + 1) % 9
					r = a % 20
					p = r / 20.0
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
	
	def visualHogs(self, startY, height, startX, width, hogs):
		im2 = im.copy()
		numOfCellsY, numOfCellsX, _ = np.shape(hogs)
		deltaY = height / numOfCellsY				#height of one cell
		deltaX = width / numOfCellsX				#width of one cell
		
		for i in range(numOfCellsX):
			x = int(startX + i * deltaX)
			cv2.line(im2, (x, startY), (x, startY + height - 1), (0,255,0))
		
		for i in range(numOfCellsX):
			y = int(startY + i * deltaY)
			cv2.line(im2, (startX, y), (startX + width - 1, y), (0,255,0))
			
		for y in range(numOfCellsY):
			mY = round(startY + y * deltaY + deltaY / 2)			
			for x in range(numOfCellsX):
				
				maxH = np.max(hogs[y, x])
				if abs(maxH) > 20:
					winkel = 0
					for h in hogs[y, x]:
						mX = round(startX + x * deltaX + deltaX / 2)
						#0 Grad = y - Achse
						m = np.matrix([mY, mX]).T
						w = -winkel / 180 * pi
						f = h * 0.4 * min(deltaX, deltaY) / maxH
						v1 = np.matrix([-f, 0, 1]).T
						v2 = np.matrix([ f, 0, 1]).T
						
						M = np.matrix( [ [ cos(w), -sin(w),  m[0] ], \
										 [ sin(w),  cos(w),  m[1] ],  \
										 [      0,       0,     1 ]  ])
						v1 = M*v1
						v2 = M*v2
						cv2.line(im2, (int(v1[1]), int(v1[0])), (int(v2[1]), int(v2[0])), (0,0,255))
						winkel = winkel + 20				
		return im2
	
	def blockNormalization(self, hogs, sizeY = 2, sizeX = 2):
		#moves a sliding box over the hogs and normalizes 
		numOfCellsY, numOfCellsX, _ = np.shape(hogs)
		tY = numOfCellsY - sizeY + 1
		tX = numOfCellsX - sizeX + 1
		t  = sizeY * sizeX * 9
		hogs2 = np.zeros([tY * tX * t])
		i = 0
		for y in range(tY):
			for x in range(tX):
				H = hogs[y:(y + sizeY), x:(x + sizeX), :]
				m = np.linalg.norm(H)
				if m > 0.0001:
					H = (1.0/m) * H
				H = H.reshape(t)
				hogs2[(i*t):(i*t + t)] = H
				i = i + 1
		return hogs2
	
if __name__ == "__main__":
	print("Start")
	ds = Dataset.Dataset("c:/here_are_the_frames", "jpg")
	im = ds.getFrame(0)
	im = cv2.imread("c:/here_are_the_frames/test/004.jpg")
	dets = Detection(im)
	#dets.ang = np.array([[80,15,0,175],[20,30,100,105]])
	#dets.mag = np.array([[4,8,12,16]  ,[0,4,8,12]])
	h, w, _ = np.shape(im)
	hogs = dets.makeHogs(0, h, 0, w, 16, 8)
	hogs2 = dets.blockNormalization(hogs, 2, 2)
	imV = dets.visualHogs(0, h, 0, w, hogs)
	cv2.imshow("Visual", imV)
	cv2.waitKey()
	print(np.shape(hogs2))
	print("OK")
	