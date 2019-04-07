#the tracker

class multiped:
	framePath  = ""
	startFrame = 0
	endFrame   = 0
	
	def __init__(self, fPath, startF, endF):
		self.framePath  = fPath
		self.startFrame = startF
		self.endFrame   = endF
	
	def info(self):
		print("The Trackers looks for Frames in Path " + self.framePath)
		print("With Startframe: " + str(self.startFrame) + " and Endframe: " + str(self.endFrame))
		
if __name__ == '__main__':
	thetracker = multiped("c:/herearetheframes/", 1, 1000)
	thetracker.info()
	
	
	
