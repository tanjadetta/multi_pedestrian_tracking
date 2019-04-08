import numpy as np
import cv2

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
	#test movePicture
	im = cv2.imread("C:/here_are_the_frames/bg.png")
	moves = [ [0,0], [30,0], [0,30], [-30,0], [0,-30], [30,30], [-30,-30], [30,-30], [-30,30] ]
	for t in moves:
		im2 = movePicture(im, *t)
		cv2.putText(im2, str(t), (0,20), 1, 2, 255)
		cv2.imshow("test", im2)
		cv2.waitKey()
	