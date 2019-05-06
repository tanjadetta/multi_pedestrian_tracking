import numpy as np
import cv2
from math import pi
from Dataset import Dataset
import Timer

def BGRtoLAB(im):
	lab_image = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
	return lab_image

def BGRtoAB(im):
	lab_image = BGRtoLAB(im)
	ab_image  = lab_image[:,:,1:3]
	return ab_image

def LABtoBGR(im):
	bgr_image = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)
	return bgr_image

def BGRtoGray(im):
	gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	s = np.shape(gray_image)
	if len(s) == 2:
		gray_image = gray_image.reshape(s[0], s[1], 1)
	return gray_image

def BGRtoHSV(im):
	hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	return hsv_image

def BGRtoH(im):
	hsv_image = BGRtoHSV(im)
	h_image = hsv_image[:,:,0]
	s = np.shape(h_image)
	if len(s) == 2:
		h_image = h_image.reshape(s[0], s[1], 1)
	return h_image

def movePicture(im, directionX, directionY):
	result = im.copy()
	h = np.shape(im)[0]
	w = np.shape(im)[1]
	
	if directionX > 0:
		#move it to the right
		result[:, directionX:w] = result[:,0:(w-directionX)]
	elif directionX < 0:
		#move it to the left
		absX = -directionX
		result[:,0:(w-absX)] = result[:,absX:w]
	if directionY < 0:
		#move it up
		absY = -directionY
		result[0:(h-absY),:] = result[absY:h,:]
	elif directionY > 0:
		#move it down
		result[directionY:h,:] = result[0:(h-directionY),:]
	return result

def clamp(x, a, b):
	return min(max(x, a), b)


def almostEqual(a,b):
	return np.abs(a - b) < 0.00000001

def magAndAngle(im):
	#img = np.float32(im) / 255.0
	img = np.float32(im) / 255.0
	# Calculate gradient 
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
	#cv2.imshow("Sobel x:", gx)
	#cv2.imshow("Sobel y:", gy)
	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
	angle = angle % 180
	#cv2.imshow("MAG:", mag)
	#cv2.imshow("Angle:", angle)
	#cv2.waitKey()
	print("max,min in angle:", np.max(angle), np.min(angle))
	print("max,min in mag:", np.max(mag), np.min(mag))
	#for each pixel chose the channel with the biggest magmitude:
	h, w, c = np.shape(im)
	mag2 = mag.reshape(h*w, c).T
	ang2 = angle.reshape(h*w, c).T
	M = np.max(mag2, axis=0)
	G = [None] * c
	for i in range(c):
		G[i] = almostEqual(mag2[i], M)			#compare ith row
		ang2[i] = ang2[i] * G[i]
	
	ang2 = np.max(ang2, axis=0)
	ang2 = ang2.reshape(h, w)
	M = M.reshape(h, w)
	
	return M, ang2

def rotatePicture(im, angle, inDegrees = True, bordercolor = None):
	s = np.shape(im)[:2]
	center = np.float32(np.array([s[0] / 2, s[1] / 2]))
	if inDegrees:
		a = angle
	else:
		a = angle * 180 / pi
	M = cv2.getRotationMatrix2D(tuple(center), a, 1.0)
	if bordercolor == None:
		return cv2.warpAffine(im, M, s, borderMode=cv2.BORDER_TRANSPARENT)
	else:
		return cv2.warpAffine(im, M, s, borderValue = bordercolor)



def plotHelper(im, titel, warte=True):
	ma = np.max(im)
	mi = np.min(im)
	r = ma - mi
	if almostEqual(r, 0):
		im2 = im - mi
	else:
		im2 = (255.0/r)*(im - mi)
	im2 = im2.astype(np.uint8)
	cv2.imshow(titel, im2)
	if warte == True:
		cv2.waitKey()
	else:
		cv2.waitKey(10)
	

#berechnet das Skalarprodukt
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

def cross2(u,v):
	return u[0]*v[1] - v[0]*u[1]
# testet, ob Punkt P auf der gleichen Seite von Strecke AB wie Punkt C liegt:
#
#    P
#
# A-------B
#        
#      C
#
#thanks to: http://blackpawn.com/texts/pointinpoly/default.html
#
#erwartet p, A, B, C als 3x1 np.arrays 
def sameSide3D(P, A, B, C):
	n1 = np.cross(B - A, P - A)
	n2 = np.cross(B - A, C - A)
	if sp(n1, n2) >= 0:  
		return True
	else: 
		return False

# testet, ob Punkt P auf der gleichen Seite von Strecke AB wie Punkt C liegt in 2D:
def sameSide2D(P, A, B, C):
	s1 = cross2(B - A, P - A)
	s2 = cross2(B - A, C - A)
	if s1*s2 >= 0:  
		return True
	else: 
		return False
	
# testet, ob P im Dreieck liegt:
#
# A------>B
#  \     /
#   \ p /
#    \ /
#     C
#
#thanks to: http://blackpawn.com/texts/pointinpoly/default.html
#
#erwartet P, A, B, C als 3x1 numpy arrays
def pointInTriangle2D(P, A, B, C):
	if sameSide2D(P, A, B, C):
		if sameSide2D(P, B, C, A):
			if sameSide2D(P, C, A, B): 
				return True
	return False

#testet, ob p im 4-Eck liegt
# A-------B
#  \       \
#	\  p    \
#    D-------C
#erwartet P, A, B, C, D als 3x1 numpy arrays
def pointIn4Eck(P, A, B, C, D):
	if pointInTriangle2D(P, A, B, C):
		return True
	if pointInTriangle2D(P, C, D, A):
		return True
	return False
			

# returns the bounding rectangle of 4 points p1, p2, p3, p4
def boundingRect(P1,P2,P3,P4):
	p1 = np.squeeze(np.asarray(P1))
	p2 = np.squeeze(np.asarray(P2))
	p3 = np.squeeze(np.asarray(P3))
	p4 = np.squeeze(np.asarray(P4))
	minY = min(p1[1], p2[1], p3[1], p4[1])
	maxY = max(p1[1], p2[1], p3[1], p4[1])
	minX = min(p1[0], p2[0], p3[0], p4[0])
	maxX = max(p1[0], p2[0], p3[0], p4[0])
	return minX, minY, maxX, maxY

# gets a mx1 matrix and returns an integer tuple
def toIntTuple(x2d):
	t = np.squeeze(np.asarray(x2d))
	t = tuple(t.astype(int))
	return t

def plotCircle(im, r, color, *xy):
	for p in xy:
		t = toIntTuple(p)
		cv2.circle(im, (t[0], t[1]), r, color)

def plotRect(im, x1, y1, x2, y2, color = (128,128,128)):
	cv2.rectangle(im, ( int(round(x1)), int(round(y1))), 
				(int(round(x2)), int(round(y2))), 
				color)
	
def plotLine(im, P1, P2):
	cv2.line(im, toIntTuple(P1), 
				toIntTuple(P2), 
				(255,255,0))
#---------------------TESTS----------------------------------

def testRotation():
	ds = Dataset("C:/here_are_the_frames/train", "jpg")
	while ds.getNextFrame():
		for angle in range(0,360,20):
			im2 = rotatePicture(ds.im, angle, True)
			plotHelper(im2, "rotated")

def testPointInTri2():
	im = np.zeros([200,200], dtype=np.uint8)
	A = np.array([100, 20])	#oben
	B = np.array([199, 150])	#unten rechts
	C = np.array([0, 180])		#unten links
	with Timer.Timer("mmmh"):
		for y in range(200):
			for x in range(200):
				p = np.array([y,x]).T
				if pointInTriangle2D(p, A, B, C):
					im[y,x] = 255
	
	plotHelper(im, "bla")
	
def testPointIn4Eck():
	im = np.zeros([200,200], dtype=np.uint8)
	A = np.array([100, 20])	 #oben
	B = np.array([199, 150]) #unten rechts
	C = np.array([0, 180]) 	#unten links
	D = np.array([10, 10])	#oben
	with Timer.Timer("mmmh"):
		for y in range(200):
			for x in range(200):
				p = np.array([x,y]).T
				if pointIn4Eck(p, A, B, C, D):
					im[y,x] = 255
	plotHelper(im, "bla")

def testBoundingRect():
	im = np.zeros([200,200,3], dtype=np.uint8)
	A = np.array([100, 20])	#oben
	B = np.array([199, 150])	#unten rechts
	C = np.array([0, 180])	#unten links
	D = np.array([10, 10])
	plotCircle(im, 3, (255,255,0), A, B, C, D)
	
	x1,y1,x2,y2 = boundingRect(A, B, C, D)
	cv2.rectangle(im, (x1,y1), (x2,y2), (255,0,0))
	plotHelper(im, "bla")
if __name__ == '__main__':
	#testRotation()
	testPointInTri2()
	testPointIn4Eck()
	testBoundingRect()
	exit()
	#test magAndAnngle
	im = cv2.imread("C:/here_are_the_frames/00000002.jpg")
	mag, angle = magAndAngle(im)
	plotHelper(mag, "MAG", False)
	plotHelper(angle, "angle", False)
	cv2.waitKey()
	print(mag)
	exit()
	#test clamp
	print(clamp(14, 10,20))
	print(clamp(4, 10,20))
	print(clamp(24, 10, 20))
	#test movePicture
	im = cv2.imread("C:/here_are_the_frames/test/001.jpg")
	moves = [ [0,0], [30,0], [0,30], [-30,0], [0,-30], [30,30], [-30,-30], [30,-30], [-30,30] ]
	for t in moves:
		im2 = movePicture(im, *t)
		cv2.putText(im2, str(t), (0,20), 1, 2, 255)
		cv2.imshow("test", im2)
		cv2.waitKey()
	