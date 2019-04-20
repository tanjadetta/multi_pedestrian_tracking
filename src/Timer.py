import time

# thanks to: 
# https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions

class Timer(object):
	def __init__(self, name=None):
		self.name = name
		
	def __enter__(self):
		self.tstart = time.time()

	def __exit__(self, type, value, traceback):
		if self.name:
			print("{0} elapsed in {1:0.2f} seconds".format(self.name, (time.time() - self.tstart)))