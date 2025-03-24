import numpy as np
# sum.py

def test():
	print('test')

def setup():
	global sa
	try:
		import SharedArray as sa
	except:
		sa = None
		return 0
	global array
	try:
		array = sa.create("shm://test5",15)
	except:
		array = sa.attach("shm://test5") 
	for i in range(15):
		array[i] = i
	print('finished shm setup')
	return 1
def getJointPos():
	global array
	#print(array)
	return array[:8]
def sendCollisionData(fk,collisions):
	global array
	#print('sending data')
	array[8] = collisions[1]
	array[9] = fk[1]
	array[10] = fk[2]
	array[11] = fk[3]
	#print(array)
	array[0:8] = 0

def sum_from_python2(a, b,array):
	return a + b + array[0]