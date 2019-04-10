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
	
if __name__ == '__main__':
	#test movePicture
	print(clamp(14, 10,20))
	print(clamp(4, 10,20))
	print(clamp(24, 10, 20))
	
	im = cv2.imread("C:/here_are_the_frames/test/001.jpg")
	moves = [ [0,0], [30,0], [0,30], [-30,0], [0,-30], [30,30], [-30,-30], [30,-30], [-30,30] ]
	for t in moves:
		im2 = movePicture(im, *t)
		cv2.putText(im2, str(t), (0,20), 1, 2, 255)
		cv2.imshow("test", im2)
		cv2.waitKey()
	