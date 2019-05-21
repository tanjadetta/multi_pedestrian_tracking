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
		

def trainClassi():
	ds = Dataset.Dataset("../data/circles/train/many/", "jpg")
	detection.Detection.train(ds)
	
	
def abcdefg():
	ds = Dataset.Dataset("../data/circles/train/many/", "jpg")
	ds.getNextFrame()
	maxH, maxW, _ = np.shape(ds.im)
	while ds.getNextFrame():
		h, w, _ = np.shape(ds.im)
		maxH = max(h, maxH)
		maxW = max(w, maxW)
	ds.aktuFrameIndex = -1
	#scale all to height max-height:
	ims = []
	while ds.getNextFrame():
		if not "pos" in ds.getFilename():
			continue
		h = np.shape(ds.im)[0]
		f = maxH / h
		im = cv2.resize(ds.im, (0,0), fx = f, fy = f, interpolation=cv2.INTER_LINEAR)
		h, w, _ = np.shape(im)
		print(h,w,maxH,maxW)
		if w < maxW:
			np.hstack((im, np.zeros([maxH, maxW - w, 3])))
		im, _ = util.magAndAngle(im)
		ims.append(im)
		#util.plotHelper(im)
	imRes = np.zeros([maxH,maxW])
	for im in ims:
		imRes = imRes + im
	
	imRes = np.log10(1 + imRes)
	util.plotHelper(imRes)
		
		
	


class SlideOnFrame:
	def __init__(self, im, width, height, svm, maske):
		cv2.namedWindow("Hallo")
		cv2.setMouseCallback("Hallo", self.mausi)
		self.cam = kamera.kamera("../data/calibration/iscam2.cali", "")
		self.h, self.w, _ = np.shape(im)
		self.width = width
		self.height = height
		self.det = detection.Detection(im, maske)
		self.svm = svm
		
	def mausi(self, event, x, y, flags, param):
		#if event == cv2.EVENT_LBUTTONDOWN:
		if event == cv2.EVENT_MOUSEMOVE:
			dstIm = im.copy()
			fpUV = np.matrix([x, y]).T
			fp = self.cam.unproj(fpUV)
			upV = np.matrix([(0,), (0,), (1,)])
			rightV = self.cam.kamkoord2weltkoord(np.matrix( [(1,), (0,), (0,)] )) -  self.cam.kamkoord2weltkoord(np.matrix( [(0,), (0,), (0,)] ))
			#rightV = np.matrix( [(1,), (0,), (0,)] )
			print("Norm von rightV", np.linalg.norm(rightV))
			rightV = (1.0 / np.linalg.norm(rightV)) * rightV
			print("rightV: ", rightV)
			v1 = fp - (self.width/2.0) * rightV
			v2 = fp + (self.width/2.0) * rightV
			v3 = v1 + self.height * upV
			v4 = v2 + self.height * upV
			imaP1 = self.cam.proj(v1).T	#row-vectors
			imaP2 = self.cam.proj(v2).T
			imaP3 = self.cam.proj(v3).T
			imaP4 = self.cam.proj(v4).T
			imaPFP = self.cam.proj(fp).T
			imaPHP = self.cam.proj(fp + upV).T
			util.plotLine(dstIm, imaPHP, imaPFP)
			util.plotLine(dstIm, imaP1, imaP2)
			util.plotCircle(dstIm, 3, (255,255,0), imaP1, imaP2)
			x1,y1,x2,y2 = util.boundingRect(imaP1, imaP2, imaP3, imaP4)
			util.plotCircle(dstIm, 3, (255,255,0), imaP1, imaP2, imaP3, imaP4)
			util.plotCircle(dstIm, 5, (255,255,255), imaPFP, imaPHP)
			util.plotRect(dstIm, x1, y1, x2, y2)
			x1 = int(x1)
			x2 = int(x2)
			y1 = int(y1)
			y2 = int(y2)
			imaP1 = np.squeeze(np.asarray(imaP1))
			imaP2 = np.squeeze(np.asarray(imaP2))
			imaP3 = np.squeeze(np.asarray(imaP3))
			imaP4 = np.squeeze(np.asarray(imaP4))
			dstH, dstW, _ = np.shape(dstIm)
			y1 = util.clamp(y1, 0, dstH - 1)
			y2 = util.clamp(y2, 0, dstH - 1)
			x1 = util.clamp(x1, 0, dstW - 1)
			x2 = util.clamp(x2, 0, dstW - 1)
			
			if y2 > y1 + 8 and x2 > x1 + 8:
				testData = np.zeros([1, 3*3*4*9])
				h = y2 - y1 + 1
				w = x2 - x1 + 1
				hogs = self.det.makeHogs(y1, h, x1, w, 4, 4)
				util.plotHelper(self.det.visualHogs(y1, h, x1, w, hogs), "bla", False)
				v = self.det.blockNormalization(hogs, 2, 2)
				testData[0] = v
				result = self.svm.predict(np.float32(testData))[1]
				print("GT  Predict  Name")
				#for i in range(ds.frameCount):
				#	print(gtLabels[i,0], "  ", result[i, 0], " ", ds.getFilename(i) )	
				print(result[0, 0])
				if result[0,0]:
					util.plotRect(dstIm, x1, y1, x2, y2, (255,0,0))
					
				#util.plotRect(dstIm, x1, y1, x2, y2, (255,0,0))	
			cv2.imshow("Hallo", dstIm)
			
	
	def slide(self):
		imRes = im.copy()
		for y in range(0, np.shape(im)[0], 4):
			for x in range(0, np.shape(im)[1], 4):
				dstIm = im.copy()
				fpUV = np.matrix([x, y]).T
				fp = self.cam.unproj(fpUV)
				upV = np.matrix([(0,), (0,), (1,)])
				rightV = self.cam.kamkoord2weltkoord(np.matrix( [(1,), (0,), (0,)] )) -  self.cam.kamkoord2weltkoord(np.matrix( [(0,), (0,), (0,)] ))
				#rightV = np.matrix( [(1,), (0,), (0,)] )
				#print("Norm von rightV", np.linalg.norm(rightV))
				rightV = (1.0 / np.linalg.norm(rightV)) * rightV
				#print("rightV: ", rightV)
				v1 = fp - (self.width/2.0) * rightV
				v2 = fp + (self.width/2.0) * rightV
				v3 = v1 + self.height * upV
				v4 = v2 + self.height * upV
				imaP1 = self.cam.proj(v1).T	#row-vectors
				imaP2 = self.cam.proj(v2).T
				imaP3 = self.cam.proj(v3).T
				imaP4 = self.cam.proj(v4).T
				imaPFP = self.cam.proj(fp).T
				imaPHP = self.cam.proj(fp + upV).T
				util.plotLine(dstIm, imaPHP, imaPFP)
				util.plotLine(dstIm, imaP1, imaP2)
				util.plotCircle(dstIm, 3, (255,255,0), imaP1, imaP2)
				x1,y1,x2,y2 = util.boundingRect(imaP1, imaP2, imaP3, imaP4)
				util.plotCircle(dstIm, 3, (255,255,0), imaP1, imaP2, imaP3, imaP4)
				util.plotCircle(dstIm, 5, (255,255,255), imaPFP, imaPHP)
				util.plotRect(dstIm, x1, y1, x2, y2)
				x1 = int(x1)
				x2 = int(x2)
				y1 = int(y1)
				y2 = int(y2)
				imaP1 = np.squeeze(np.asarray(imaP1))
				imaP2 = np.squeeze(np.asarray(imaP2))
				imaP3 = np.squeeze(np.asarray(imaP3))
				imaP4 = np.squeeze(np.asarray(imaP4))
				dstH, dstW, _ = np.shape(dstIm)
				y1 = util.clamp(y1, 0, dstH - 1)
				y2 = util.clamp(y2, 0, dstH - 1)
				x1 = util.clamp(x1, 0, dstW - 1)
				x2 = util.clamp(x2, 0, dstW - 1)
				
				if y2 > y1 + 8 and x2 > x1 + 8:
					testData = np.zeros([1, 3*3*4*9])
					h = y2 - y1 + 1
					w = x2 - x1 + 1
					hogs = self.det.makeHogs(y1, h, x1, w, 4, 4)
					#util.plotHelper(self.det.visualHogs(y1, h, x1, w, hogs), "bla", False)
					v = self.det.blockNormalization(hogs, 2, 2)
					testData[0] = v
					result = self.svm.predict(np.float32(testData))[1]
					#print("GT  Predict  Name")
					#for i in range(ds.frameCount):
					#	print(gtLabels[i,0], "  ", result[i, 0], " ", ds.getFilename(i) )	
					#print(result[0, 0])
					if result[0,0]:
						util.plotRect(imRes, x1, y1, x2, y2, (255,0,0))
						print ("!!!!!!!!!!!!!!!!!!!!!!!!!")
						
					#util.plotRect(dstIm, x1, y1, x2, y2, (255,0,0))
					#util.plotHelper(dstIm, "In Bearbeitung", False)
			print (y)	
		util.plotHelper(imRes, "Im Res")
	


	
if __name__ == '__main__':
	#makeManyTrainPics()
	#projectTestsOnFrame()
	#trainClassi()
	#abcdefg()
	#exit()
	svm = detection.Detection.loadSVM("model.dat")
	im = cv2.imread("c:/here_are_the_frames/00test.jpg")
	maske = cv2.imread("../data/circles/maske.jpg")
	so = SlideOnFrame(im, 1.0, 1.0, svm, maske)
	#so.slide()
	#exit()
	while(True):
		k = cv2.waitKey(20)
		if k == 27:
			break