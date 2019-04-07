#the tracker
import cv2

import Dataset

class Multiped:
	dataset = None
	
	def __init__(self, ds):
		ds.info()
		self.dataset = ds
	
	def info(self):
		return "The Tracker uses the following Dataset: " + self.dataset.info()
		
if __name__ == '__main__':
	ds = Dataset.Dataset("c:/here_are_the_frames/", "jpg")
	thetracker = Multiped(ds)
	print (thetracker.info())
	
	
	
