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
	return gray_image

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
	
if __name__ == '__main__':
	#test BGRtoLAB, LABtoBGR, BGRtoGray
	im = cv2.imread("C:/here_are_the_frames/bg.png")
	cv2.imshow("BGR", im)
	cv2.waitKey()
	imGray = BGRtoGray(im)
	cv2.imshow("GRAY", imGray)
	cv2.waitKey()
	imLAB = BGRtoLAB(im)
	im2   = LABtoBGR(imLAB)
	cv2.imshow("From BGR to LAB to BGR", im2)
	
	#test movePicture
	im = cv2.imread("C:/here_are_the_frames/bg.png")
	moves = [ [0,0], [30,0], [0,30], [-30,0], [0,-30], [30,30], [-30,-30], [30,-30], [-30,30] ]
	for t in moves:
		im2 = movePicture(im, *t)
		cv2.putText(im2, str(t), (0,20), 1, 2, 255)
		cv2.imshow("test", im2)
		cv2.waitKey()
	