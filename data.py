import os
import numpy as np
import shutil
from shutil import copyfile
import pickle



def create_exposures(BASE='cifar10', No_elements=200, No_Exposures=100):
	DATA_DIR = BASE+'_data/train'

	if BASE is 'cifar10':
		classes = ['plane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	elif BASE is 'mnist' or BASE is 'fashion_mnist' or BASE is 'svhn' or BASE is 'imagenette':
		classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
	elif BASE is 'cifar100':
		classes = ['streetcar', 'apple', 'palm_tree', 'man', 'forest', 'butterfly', 'lamp', 'wardrobe', 'pear', 'raccoon', 'crab', 'tractor', 'aquarium_fish', 'can', 'tank', 'snail', 'bed', 'cup', 'television', 'turtle', 'boy', 'mushroom', 'bee', 'fox', 'willow_tree', 'couch', 'dolphin', 'cattle', 'maple_tree', 'plain', 'bear', 'bridge', 'leopard', 'hamster', 'lawn_mower', 'lobster', 'camel', 'tiger', 'road', 'whale', 'motorcycle', 'crocodile', 'dinosaur', 'chair', 'bus', 'plate', 'otter', 'rose', 'seal', 'telephone', 'mouse', 'tulip', 'porcupine', 'beaver', 'wolf', 'lizard', 'flatfish', 'beetle', 'chimpanzee', 'poppy', 'bowl', 'table', 'shrew', 'skyscraper', 'kangaroo', 'cloud', 'girl', 'worm', 'train', 'house', 'caterpillar', 'spider', 'rabbit', 'cockroach', 'rocket', 'castle', 'bicycle', 'baby', 'mountain', 'sweet_pepper', 'orchid', 'sea', 'skunk', 'oak_tree', 'squirrel', 'ray', 'bottle', 'pickup_truck', 'keyboard', 'lion', 'clock', 'trout', 'orange', 'woman', 'shark', 'sunflower', 'pine_tree', 'snake', 'elephant', 'possum']

	#Create Exposures

	no_available = {}

	for class_element in classes:
		classElementPath = os.path.join(BASE+'_data/train/', class_element)
		if class_element not in no_available:
			no_available[class_element] = len(os.listdir(classElementPath))
			
	
	#Prevent Repeated Exposures

	allow_replace = False

	totalFileList = {}
	rand_exposures = []

	tempList = list(range(len(classes)))*int(No_Exposures//len(classes))
	np.random.shuffle(tempList)

	prev = tempList[0]
	rand_exposures.append({prev:No_elements})

	prob_mixed = 0
	prev_mixed = False
	for i in range(1, len(tempList)):
		if np.random.random()<prob_mixed and i>1 and i < len(tempList)-1 and not prev_mixed:
			num_first = int((np.random.random()*0.333 + 0.333) * No_elements)
			rand_exposures.append({prev:num_first, tempList[i+1]:No_elements - num_first})
			prev_mixed = True
		else:
			rand_exposures.append({tempList[i]: No_elements})
			prev_mixed = False
		if i >= len(tempList):
			break

		prev = tempList[i]
    
    
	rand_exposures = rand_exposures[:No_Exposures]
	for i in range(len(classes)):
		occurence = 0
		for j in range(len(rand_exposures)):
			if i in list(rand_exposures[j].keys()):
				occurence += rand_exposures[j][i]

		if allow_replace == False:
			if occurence > int(no_available[classes[i]]):
				print("Not enough unique samples for exposure")
				print(str(classes[i]) + " has " + str(no_available[classes[i]]) + " unique samples")
				print("Attempted to create " + str(occurence) + " samples")
			else:
				fileIndex = np.random.choice(no_available[classes[i]], occurence, replace=False)
				classElementPath = os.path.join(DATA_DIR, classes[i])
				exposureFiles = np.asarray(os.listdir(classElementPath))[fileIndex]
				totalFileList[classes[i]] = exposureFiles

		else:
			fileIndex = np.random.choice(no_available[classes[i]], occurence, replace=True)
			classElementPath = os.path.join(DATA_DIR, classes[i])
			exposureFiles = np.asarray(os.listdir(classElementPath))[fileIndex]
			totalFileList[classes[i]] = exposureFiles
			
	EXPOSURE_DIR = 'exposures_mixed_'+BASE

	indexMap = {}
	for element in classes:
		indexMap[element] = 0


	if os.path.isdir(EXPOSURE_DIR):
		shutil.rmtree(EXPOSURE_DIR)
	
	os.mkdir(EXPOSURE_DIR)
		
	exposureMap = {}

	for i in range(len(rand_exposures)):

		exposureElementDir = os.path.join(EXPOSURE_DIR,str(i))
		if not os.path.isdir(exposureElementDir):
			os.mkdir(os.path.join(exposureElementDir))  
		classIDs = rand_exposures[i]
		exposureMap[i]  = []
		for key, value in sorted(classIDs.items(), key=lambda item: item[1], reverse=True):
			exposureMap[i].append((classes[key], key))
		for classID in list(classIDs.keys()):
			startingIndex = int(indexMap[classes[classID]])

			expElementFiles = totalFileList[classes[classID]][startingIndex:startingIndex + classIDs[classID]]   
			indexMap[classes[classID]] += classIDs[classID]

			for file in expElementFiles:

				src = os.path.join(os.path.join(DATA_DIR, classes[classID]),file)
				dest = os.path.join(exposureElementDir, file)

				copyfile(src, dest)
	
	pickle.dump(exposureMap, open("exposureMap_mixed_"+BASE+".pkl", "wb" ))
	
	return