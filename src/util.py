import numpy as np
import cv2

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
	cv2.imshow("MAG:", mag)
	cv2.imshow("Angle:", angle)
	cv2.waitKey()
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
	
if __name__ == '__main__':
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
	