#the tracker

import Dataset

class multiped:
	dataset = None
	
	def __init__(self, ds):
		ds.info()
		self.dataset = ds
	
	def info(self):
		return "The Trackers uses the following Dataset: " + self.dataset.info()
		
if __name__ == '__main__':
	ds = Dataset.Dataset("c:/here_are_the_frames/", 1, 1000)
	thetracker = multiped(ds)
	print (thetracker.info())
	
	
	
