class Dataset:
	path = ""
	startFrame = 0
	endFrame   = 0
	
	def __init__(self, path, startF, endF):
		self.path = path
		self.startFrame = startF
		self.endFrame = endF

	def info(self):
		return "Dataset with \n" \
		"path = "       + str(self.path) + "\n"  \
		"startFrame = " + str(self.startFrame) + "\n" \
		"endFrame = " + str(self.endFrame)

if __name__ == '__main__':
	ds = Dataset("c:/herearetheframes/", 1, 1000)
	print (ds.info())