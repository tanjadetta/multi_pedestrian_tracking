import numpy as np
from xml.dom import minidom
from math import acos
from math import copysign
import util
import cv2
import Dataset
from _testbuffer import staticarray
from util import plotHelper

class kamera:
	
	def __init__(self, dateiName, welteinheit):
		#self.xmlDatei = xmlDatei
		if dateiName[-5:].upper() == ".CALI":
			self.ladeAusDatei2(dateiName)
		elif dateiName[-4:].upper() == ".XML":
			self.ladeAusDatei(welteinheit, dateiName)
		else:
			raise NameError('Hello World!')
	def getKamPos(self):
		return -self.R.T*self.T
	
	def rekalk(self):
		self.P = kamera.getWeltMatrix(self.R, self.T, self.K)
		self.H = kamera.getHomo(self.P)
		self.Hinv = np.linalg.inv(self.H)
		
	def setzeOrigo1(self, px, py):
		#Setze Welt-Origo auf 100,100:
		#Da T die Position des Origos is Kam-Koordinaten ist, ist KT die Position des Origos in Pixelkoordinaten
		#Setze KT auf (px,py)
		#KT = (px,py,1)
		#<=> T = K^-1 * (px,py,1)
		#print("Altes T", self.T)
		T = np.linalg.inv(self.K) * np.matrix([px,py,1]).T
		#print("vorlaufiges T")
		#print(T)
		#Aber: das T ist "up to scale"
		#das heisst jedes sT liefert das Ergebnis
		#Habe das Constraint c_z = altes c_z
		C = self.getKamPos()
		cz = C[2,0]
		#Es gilt C = -R.T sT =: v * s
		v = -self.R.T * T
		#loese nach s auf, mittels letzter Zeile: cz = v_3 * s
		s = cz / v[2,0]
		T = s*T
		#print("Neues T")
		#print(T)
		self.T = T
		self.rekalk()
	
	def rotaZ(self):
		#rotiere Kamera, also C um die Welt-Z-Achse um ein Grad
		w = 1.0 / 180 * np.pi
		c, s = np.cos(w), np.sin(w)
		RZ = np.matrix([ (c,-s,0),
						(s, c,0),
						(0, 0,1) ])
		self.R = self.R * RZ
		self.rekalk()
		
	def saveas(self):
		x = input("Dateiname? ")
		if len(x):
			np.savetxt(x+".cali.R", self.R)
			np.savetxt(x+".cali.K", self.K)
			np.savetxt(x+".cali.T", self.T)
		
	def ladeAusDatei2(self, datNam):
		R = np.asmatrix(np.loadtxt(datNam+".R"))
		K = np.asmatrix(np.loadtxt(datNam+".K"))
		T = np.asmatrix(np.loadtxt(datNam+".T")).T	#muss .T
		self.R = R
		self.T = T
		self.K = K
		#print("K")
		#print(K)
		#print("R")
		#print(R)
		#print("T")
		#print(T)
		self.P = kamera.getWeltMatrix(R, T, self.K)
		self.H = kamera.getHomo(self.P)
		self.Hinv = np.linalg.inv(self.H)
		
	def ladeAusDatei(self, welteinheit, datNam):
		#datNam = self.xmlDatei
		xmldoc = minidom.parse(datNam)
		item = xmldoc.getElementsByTagName('Geometry')[0]
		dpx = float(item.attributes['dpx'].value)
		if "dx" in item.attributes:
			if float(item.attributes['dx'].value) > 0.00000000001:
				dpx = float(item.attributes['dx'].value)
		dpy = float(item.attributes['dpy'].value)
		width = float(item.attributes['width'].value)
		height = float(item.attributes['height'].value)
		print("dpx = ",dpx)
		print("dpy = ",dpy)
		print("width = ",width)
		print("height = ",height)
		item = xmldoc.getElementsByTagName('Intrinsic')[0]
		f = float(item.attributes['focal'].value)
		cx = float(item.attributes['cx'].value)
		cy = float(item.attributes['cy'].value)
		print("f = ",f)
		print("cx = ",cx)
		print("cy = ",cy)
		item = xmldoc.getElementsByTagName('Extrinsic')[0]
		tx = float(item.attributes['tx'].value)
		ty = float(item.attributes['ty'].value)
		tz = float(item.attributes['tz'].value)
		if welteinheit == "mm":
			tx,ty,tz = tx/1000, ty/1000, tz/1000
		rx = float(item.attributes['rx'].value)
		ry = float(item.attributes['ry'].value)
		rz = float(item.attributes['rz'].value)
		print("rx=",rx,"  ry=",ry, "  rz=",rz)
		T = np.matrix(((tx,),(ty,),(tz,)))
		print(T)
		c = np.cos(rx)
		s = np.sin(rx)
		RX = np.matrix([ (1,0, 0), 
						(0,c,-s), 
						(0,s, c) ])
		print(RX)
		c = np.cos(ry)
		s = np.sin(ry)
		RY = np.matrix([ (c, 0,s), 
						(0, 1,0),  
						(-s,0,c) ])
		print(RY)
		c, s = np.cos(rz), np.sin(rz)
		RZ = np.matrix([ (c,-s,0),
						(s, c,0),
						(0, 0,1) ])
		print(RZ)
		R = RZ*RY*RX
		
		self.R = R
		self.T = T
		self.K = kamera.getKameraMatrix(f, dpx, dpy, cx, cy)
		self.P = kamera.getWeltMatrix(R, T, self.K)
		self.H = kamera.getHomo(self.P)
		self.Hinv = np.linalg.inv(self.H)
		
	@staticmethod
	def getKameraMatrix(f, dpx, dpy, cx, cy):				#innere Orientierung
		K = np.matrix( [(f*(1.0/dpx), 0			 , cx),
						(0         	, f*(1.0/dpy), cy),
						(0			, 0			 , 1)] )
		print(K)
		return K

	@staticmethod
	def getWeltMatrix(R, T, K):
		P = K * np.hstack((R,T))
		return P
	
	@staticmethod
	def unHom(X):
		#print("HOM",X)
		X2 = (1.0/X[-1,0]) * X
		#print("Geteilt durch letzte:",X2)
		#print("kamera.unHom", X2[0:-1,0])
		return X2[0:-1,0]

	@staticmethod
	def toHom(X):		#haenge eine 1 unten ran
		x2 = np.vstack((X,np.matrix( ((1,)) )  ))
		return x2

	@staticmethod
	def to2dMitMatrix(P,X):			#proj. auf die Bildebene
		x3 = P * kamera.toHom(X)
		return kamera.unHom(x3)
		
	@staticmethod
	def to2dMitH(H,X):
		#Eingabe: 3D-Punkt X=[X,Y,0]
		#Ausgabe: uv Koordinaten
		x2 = X.copy()
		x2[2,0] = 1
		return kamera.unHom(H * x2)
	
	@staticmethod
	def to3D(uv, H):
		#Eingabe ein Punkt auf der Bild-Ebene, Ausgabe: 3D-Koordinate (genauer (X,Y)-Koordinate)
		#print("H",H)
		#print("uv", uv)
		Hinv = np.linalg.inv(H)
		#uvh = kamera.toHom(uv)
		#print("Hinv", Hinv)
		#print("uv in homog", uvh)
		#print("Hinv*uvh", Hinv * uvh)
		return np.vstack((kamera.unHom(Hinv * kamera.toHom(uv)), np.matrix(((0,)))))

	def proj(self, X):
		#Eingabe: ein 3D Punkt X = (X,Y,Z)
		#Ausgabe: Pixel-Koordinate (uv)
		return kamera.to2dMitMatrix(self.P, X)
	
	def unproj(self, uv):
		#Eingabe: uv-Pixelkoordinate
		#Ausgabe: Weltkoordinate = (X,Y,0)
		return kamera.to3D(uv, self.H)
	def kampos(self):
		C = -self.R.T * self.T
		print("Kameraposition:", C)
		return C
	
	@staticmethod
	def getHomo(P):
		#P ist die 3x4 Weltmatrix (Projektionsmatrix)
		#streiche 3.spalte in P (da z immer=0)
		H =  np.hstack( (P[:,0:2], P[:,3] )  )
		return H
	
	@staticmethod
	def getHomoWand(P):
		#P ist die 3x4 Weltmatrix (Projektionsmatrix)
		#streiche 2.spalte in P (da y immer=0)
		H =  np.hstack( (P[:,0], P[:,2:] )  )
		return H
	
	
	def kamkoord2weltkoord(self, XK):
		#Eingabe: Punkt XK in Kamera-Koordinaten 
		#Ausgabe: Koordinate in Weltkoordinaten
		return self.R.T*(XK - self.T)
	
	@staticmethod
	def sp(v, w):
		v2 = np.asmatrix(v)
		w2 = np.asmatrix(w)
		#print (np.shape(v2))
		#print (np.shape(w2))
		if np.shape(v2)[0] == 1 and np.shape(w2)[0] == 1:
			s = v2 * w2.T
		elif np.shape(v2)[1] == 1 and np.shape(w2)[1] == 1:
			s = v2.T * w2
		else: 
			raise NameError('HiThere')
		assert(np.shape(s) == (1, 1))
		return s[0,0]
	
	@staticmethod
	def winkel(v, w):
		#Eingabe: zwei Richtungsvektoren
		#Ausgabe: Winkel zwischen den beiden Vektoren
		s = kamera.sp(v, w) 
		#if s < 0:
		#	s = -s
		return (acos(s / (np.linalg.norm(v) * np.linalg.norm(w))  ))
	
	@staticmethod
	def tiefe(x3d, R, T, K):
		P = kamera.getWeltMatrix(R, T, K)
		M = P[:, 0:-1]
		x4d = kamera.toHom(x3d)
		xproj = P*x4d
		d = np.linalg.det(M)
		#print(xproj)
		w = xproj[2,0]
		return copysign(1, d)*w / np.linalg.norm(M[2,:])
	
	
	
	@staticmethod
	def findHomoOfRectangle(width, height, imagePoints):
		#width and height are floats 
		#imagePoints should be have the form 4 x 2 .
		#Finds the homography between the plane induced of the rectangle in the world
		#and the image plane where pS are the projected corners of that
		#rectangle
		
		imaPoints = np.float32(imagePoints.copy())
		wPoints   = np.float32([[0    ,      0] ,
							    [width,      0], \
								[0    , height], 
								[width, height]])
		
		H, _ = cv2.findHomography(wPoints, imaPoints, 0) 
	
		Probe = H * np.asmatrix((np.hstack( (wPoints, np.matrix([1,1,1,1]).T) )).T)
		for i in range(4):
			Probe[0,i] = Probe[0,i] / Probe[2,i]
			Probe[1,i] = Probe[1,i] / Probe[2,i]
			Probe[2,i] = Probe[2,i] / Probe[2,i]
		
		print("Imapoints:" )
		print(imaPoints)
		print("Probe:")
		print(np.round(Probe.T))
		
		return H

	
	def projectPicture(self, fp, width, height, srcIm, dstIm):
		#projects a picture in a vertical to the x-y-plane and 
		#towards the camera looking rectangle 
		#onto the image plane
		#the footPoint should be a 3x1 vector of the form (x,y,z).T 
		upV = np.matrix([(0,), (0,), (1,)])
		rightV = self.kamkoord2weltkoord(np.matrix( [(1,), (0,), (0,)] )) -  self.kamkoord2weltkoord(np.matrix( [(0,), (0,), (0,)] ))
		#rightV = np.matrix( [(1,), (0,), (0,)] )
		rightV = (1.0 / np.linalg.norm(rightV)) * rightV
		print("Norm von rightV", np.linalg.norm(rightV))
		print("rightV: ", rightV)
		v1 = fp - (width/2.0) * rightV
		v2 = fp + (width/2.0) * rightV
		v3 = v1 + height * upV
		v4 = v2 + height * upV
		imaP1 = self.proj(v1).T	#row-vectors
		imaP2 = self.proj(v2).T
		imaP3 = self.proj(v3).T
		imaP4 = self.proj(v4).T
		imaPFP = self.proj(fp).T
		imaPHP = self.proj(fp + upV).T
		util.plotLine(dstIm, imaPHP, imaPFP)
		util.plotLine(dstIm, imaP1, imaP2)
		util.plotCircle(dstIm, 3, (255,255,0), imaP1, imaP2)
		imagePoints = np.vstack( (imaP1, imaP2, imaP3, imaP4) )
		H = kamera.findHomoOfRectangle(width, height, imagePoints)
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
		for y in range(y1,y2+1):
			for x in range(x1,x2+1):
				p = np.array([x,y])
				if util.pointIn4Eck(p, imaP1, imaP2, imaP4, imaP3):
					dstIm[y,x,2] = 255
				
def testProjPic():
	kam = kamera("c:/here_are_the_frames/calibration/iscam2.CALI", "")
	dstIm = cv2.imread("c:/here_are_the_frames/00000002.jpg")
	srcIm = cv2.imread("C:/here_are_the_frames/test2/pos_004.jpg")
	fp = np.matrix([0,0,0]).T
	kam.projectPicture(fp, 1, 1, srcIm, dstIm)
	util.plotHelper(dstIm, "Ziel", False)
	util.plotHelper(srcIm, "Source")
	
	
	

if __name__ == '__main__':
	testProjPic()
	#testRotation()
	exit()
		
		