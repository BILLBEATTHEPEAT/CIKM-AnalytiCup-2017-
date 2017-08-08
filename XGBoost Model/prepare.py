### This code is for data split and format transformation


import numpy as np 


def loadfile(f):

	if f == 0:
		fr = open('/home/CIKM2017/CIKM2017_train/train.txt')
	elif f == 1:
		fr = open('/home/CIKM2017/CIKM2017_test/test.txt')

	return fr

def lineSplit(line):
	sample = line.strip().split(' ')
	label = float(sample[0].split(',')[1])

	first_part_split = sample[0].split(',')[2]
	sample = [first_part_split] + sample[1:]
	sample = [float(i) for i in sample]
	sample = np.array(sample).reshape(15,4,101,101)

	return sample,label

def main():

	print "split the train data"

	dataArr = []
	YArr = []
	count = 0
	fr = loadfile(0)

	for l in fr:

		print count

		data, y = lineSplit(l)

		dataArr = np.append(dataArr,data[::,1,::,::])
		YArr.append(y)
		del data, l

		count += 1
	del fr

	np.save("train_only_H1.npy", dataArr)
	np.save("label.npy", YArr)

	del dataArr, YArr

	print "split the test data"

	dataArr = []
	YArr = []
	count = 0
	fr = loadfile(1)

	for l in fr:

		print count

		data, y = lineSplit(l)

		dataArr = np.append(dataArr,data[::,1,::,::])
		# YArr.append(y)
		count += 1
	del fr

	np.save("test_only_H1.npy", dataArr)
	# np.save("raw_label.npy", YArr)

	del dataArr

if __name__ == "__main__":
	main()
		

