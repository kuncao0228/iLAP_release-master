
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pickle
import numpy as np
import copy
from TransformedDataset import TransformedDataset
import util
import ExposureHandler as ExposureHandler

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
import argparse
from data import create_exposures


def runIncremental(classNameList, pickleMap, EXPOSURE_DIR, TEST_DIR, EPOCHS = 15, DIFF_HIGH_THRESHOLD=.3, BATCH_SIZE = 16, FORGET_RATIO=.5, \
	NEW_CLASS_THRES=0.5, DIFF_LOW_THRESHOLD=0.1, DELTA_THRESHOLD = .3, MIN_CLASS_THRESHOLD=0.2, EXEMPLAR_SIZE=200, PATH="mnist_50_imba/", test=False):

	exposureList = np.array(range(len(os.listdir(EXPOSURE_DIR))))

	#If shuffled training
	if test:
		np.random.shuffle(exposureList)
		while (pickleMap[exposureList[0]] == pickleMap[exposureList[1]]):
			np.random.shuffle(exposureList)
	

	exposureMapping = []

	print("Creating Base Exposure")
	exposeHandler, model = createBaseExposures(EXPOSURE_DIR, TEST_DIR, pickleMap, exposureList, BATCH_SIZE, EPOCHS, EXEMPLAR_SIZE)
	valLoader = exposeHandler.getValidationSet()
# 	print(exposeHandler.getMapping())
	prev_accuracy = util.assessModel(model, valLoader, verbal=False)
	print("Base Exposure Accuracy is ")
	print(prev_accuracy)

	EPS = 1e-4


	for i in range(1, len(pickleMap)):

		print("*************************Running Exposure No " + str(i) +":"+ str(exposureList[i]) + " ************************")
		print("New exposure is " + pickleMap[exposureList[i]][0][0])

		previousExposureObj = copy.deepcopy(exposeHandler)
		previous_Model = copy.deepcopy(model)

		dataset = exposeHandler.queryNextExposure(exposureList[i])
		exposeHandler.updateModel(model)
		exposeHandler.updateDataSet(dataset)

        # Getting loaders for feedback analyzer
		custom_dataloader_train, custom_dataloader_val, =\
                exposeHandler.getFeedbackLoaders(FORGET_RATIO, BATCH_SIZE)
        
		exposeHandler.updateMapping(pickleMap[exposureList[i]])
		class_weights = exposeHandler.getWeights()

        # Training the feedback analyzer
		model = util.TrainModel(custom_dataloader_train, custom_dataloader_val,\
		 			len(exposeHandler.getMapping()), model, firstFlag=False, TOTAL_EPOCHS=EPOCHS, class_weights=torch.from_numpy(class_weights).float())
        
        
		exposeHandler.addValidation(len(exposeHandler.getMapping())-1,exposureList[i])
        
        #Assessing the model
        
		valLoader = exposeHandler.getValidationSet()

		accuracy = util.assessModel(model, valLoader, verbal=False)

		print("IS SEEN COMPARISON")
		print("Previous Accuracy is: " + str(prev_accuracy))
		print("Current Accuracy is: " + str(accuracy))

        # Accuracy Delta calculations
		accuracyDelta = np.asarray(accuracy[:-1])/(np.asarray(prev_accuracy)+EPS)

		deltaCount = len([acc for acc in accuracyDelta if acc < DELTA_THRESHOLD]) 

		print("Accuracy Delta Is: " +str(accuracyDelta))
		print("Accuracy Delta Threshold Exceeded Count: " + str(deltaCount))

        

# If most recent exposure learnt has less accuracy than new class threshold or multiple classes 
# drop in accuracy, then dont make decision and add to unsupervised exemplar set
		if deltaCount > 1:
			model = previous_Model
			exposeHandler = previousExposureObj
			exposeHandler.updateModel(model)
			continue
        
        # Find if class is seen before on two different thresholds
		isSeen_high, seen_index_high = util.classIsSeen(prev_accuracy, accuracy[:-1], DIFF_HIGH_THRESHOLD)
		isSeen_low, _ = util.classIsSeen(prev_accuracy, accuracy[:-1], DIFF_LOW_THRESHOLD)
        
        # If decision boundary is unclear then dont make decision and add to unsupervised exemplar set
		if isSeen_high and not isSeen_low:
			print("is Seen and not isSeen_low")
			model = previous_Model
			exposeHandler = previousExposureObj
			exposeHandler.updateModel(model)
			continue

        # If exposure is a new class
		if isSeen_high == False:
            # Going back to previous model to avoid artifacts created by class imbalance training
			model = previous_Model
			class_weights = exposeHandler.getWeights()
            
            # Get Train Loaders
			custom_dataloader_train, custom_dataloader_val =\
                exposeHandler.getTrainLoadersNew(BATCH_SIZE)
            
            # Train model with new class
			model = util.TrainModel(custom_dataloader_train, custom_dataloader_val, \
                                        len(exposeHandler.getMapping()), model, firstFlag=False, TOTAL_EPOCHS=EPOCHS, class_weights = torch.from_numpy(class_weights).float())
            
			valLoader = exposeHandler.getValidationSet()
			accuracy_new = util.assessModel(model, valLoader, verbal=False)
            
            # If new class learnt does not have high enough accuracy, dont update model
			if accuracy_new[-1] < NEW_CLASS_THRES:
				model = previous_Model
				exposeHandler = previousExposureObj
				exposeHandler.updateModel(model)
				continue
            
			prev_accuracy = accuracy_new

        # If it is old class
		else:
			print(classNameList[exposeHandler.getMapping()[seen_index_high][0][1]] + "*******EXPOSURE SEEEENNNNNNNN****")
			model = previous_Model
			exposeHandler = previousExposureObj
			seenDataSet = exposeHandler.queryNextExposure(exposureList[i], assignedLabel=seen_index_high)
			class_weights = copy.deepcopy(exposeHandler.getWeights())
			class_weights[seen_index_high]-=0.5

			custom_dataloader_train, custom_dataloader_val =\
                exposeHandler.getTrainLoadersOld(seenDataSet, BATCH_SIZE)
            
			model = util.TrainModel(custom_dataloader_train, custom_dataloader_val,\
						len(exposeHandler.getMapping()), model, firstFlag=False, TOTAL_EPOCHS=EPOCHS, class_weights=torch.from_numpy(class_weights).float())


			exposeHandler.updateModel(model)
			exposeHandler.updateDataSet(seenDataSet)
			valLoader = exposeHandler.getValidationSet()
			accuracy_new = util.assessModel(model, valLoader, verbal=False)
			prev_accuracy = accuracy_new
			
		while np.min(prev_accuracy) < MIN_CLASS_THRESHOLD:
			class_delete = np.argmin(prev_accuracy)
			exposeHandler.removeClass(class_delete, BATCH_SIZE)
			prev_accuracy[class_delete] = copy.deepcopy(prev_accuracy[-1])
			prev_accuracy.pop()
			print(exposeHandler.getMapping())
			print(prev_accuracy)



	os.makedirs(PATH, exist_ok=True)

	pickle.dump(exposeHandler.getMapping(),open(os.path.join(PATH, "modelLearnedMapping.pkl"), "wb"))
	torch.save(model, os.path.join(PATH, "final_model.pth"))
	return


def runIncrementalIolfcv(classNameList, pickleMap, EXPOSURE_DIR, TEST_DIR, EPOCHS = 15, DIFF_HIGH_THRESHOLD=.5, BATCH_SIZE = 16, FORGET_RATIO=.5, \
	NEW_CLASS_THRES=0.5, DIFF_LOW_THRESHOLD=0.1, DELTA_THRESHOLD = .5, EXEMPLAR_SIZE=300, PATH="mnist_50_imba/", test=False):

	exposureList = np.array(range(len(os.listdir(EXPOSURE_DIR))))
	if test:
		np.random.shuffle(exposureList)
		while (pickleMap[exposureList[0]] == pickleMap[exposureList[1]]):
			np.random.shuffle(exposureList)
	
	exposureMapping = []

	print("Creating Base Exposure")
	exposeHandler, model = createBaseExposures(EXPOSURE_DIR, TEST_DIR, pickleMap, exposureList, BATCH_SIZE, EPOCHS, EXEMPLAR_SIZE, num_exp=2)
	valLoader = exposeHandler.getValidationSet()
	print(exposeHandler.getMapping())
	prev_accuracy = util.assessModel(model, valLoader, verbal=False)
	print("Base Exposure Accuracy is ")
	print(prev_accuracy)

	custom_dataloaders_exemplars, __ =\
                exposeHandler.getDistanceLoaders(None, BATCH_SIZE)
	mean_distances = []
	mean_features = []
	for dataloader_exemplar in custom_dataloaders_exemplars:
		mean_distances.append(np.mean(util.getDistances(model, dataloader_exemplar),axis=0))
		mean_features.append(util.getDistances(model, dataloader_exemplar))
	distances = np.expand_dims(np.array(mean_distances),1)
	distances_first = np.mean(np.linalg.norm(distances-mean_features[0], axis=2), axis=1)
	distances_second = np.mean(np.linalg.norm(distances-mean_features[1], axis=2), axis=1)
	dist_threshold = (np.min(distances_first) + np.min(distances_second))/2
	print(dist_threshold)


	EPS = 1e-4

	for i in range(2, len(pickleMap)):
		print("*************************Running Exposure No " + str(i) +":"+ str(exposureList[i]) + " ************************")
		print("New exposure is " + pickleMap[exposureList[i]][0][0])
		previousExposureObj = copy.deepcopy(exposeHandler)
		previous_Model = copy.deepcopy(model)

		dataset = exposeHandler.queryNextExposure(exposureList[i])
		exposeHandler.updateModel(model)

        # Getting loaders for exemplar distances
		custom_dataloaders_exemplars, custom_dataloader_new =\
                exposeHandler.getDistanceLoaders(dataset, BATCH_SIZE)
        
		#Calculating distances
		distances = []
		for dataloader_exemplar in custom_dataloaders_exemplars:
			distances.append(np.mean(util.getDistances(model, dataloader_exemplar),axis=0))
		distance_new_exemplar = util.getDistances(model, custom_dataloader_new)

		distances = np.expand_dims(np.array(distances),1)
		distances = np.mean(np.linalg.norm(distances-distance_new_exemplar, axis=2), axis=1)
		min_dist = np.min(distances)
		min_class = np.argmin(distances)
		
		if min_dist < dist_threshold:
			isSeen_high = True
			seen_index_high = min_class
		else:
			isSeen_high = False
			seen_index_high = min_class
        
        

        # If exposure is a new class
		if isSeen_high == False:
			model = previous_Model
			exposeHandler.updateDataSet(dataset)
        
			exposeHandler.updateMapping(pickleMap[exposureList[i]])
			class_weights = exposeHandler.getWeights()
            
            # Get Train Loaders
			custom_dataloader_train, custom_dataloader_val =\
                exposeHandler.getTrainLoadersNew(BATCH_SIZE)
            
            # Train model with new class
			model = util.TrainModel(custom_dataloader_train, custom_dataloader_val,\
                                        len(exposeHandler.getMapping()), model, firstFlag=False, TOTAL_EPOCHS=EPOCHS, class_weights = torch.from_numpy(class_weights).float())
			#exposeHandler.addValidation(len(exposeHandler.getMapping())-1, i)
			exposeHandler.addValidation(len(exposeHandler.getMapping())-1,exposureList[i])
			
			valLoader = exposeHandler.getValidationSet()
			accuracy_new = util.assessModel(model, valLoader, verbal=False)
            
            # If new class learnt does not have high enough accuracy, dont update model
			if accuracy_new[-1] < NEW_CLASS_THRES:
				model = previous_Model
				exposeHandler = previousExposureObj

				exposeHandler.updateModel(model)


				continue
            
			prev_accuracy = accuracy_new

        # If it is old class
		else:
			print(classNameList[exposeHandler.getMapping()[seen_index_high][0][1]] + "*******EXPOSURE SEEEENNNNNNNN****")
			model = previous_Model
			exposeHandler = previousExposureObj
			seenDataSet = exposeHandler.queryNextExposure(exposureList[i], assignedLabel=seen_index_high)
			class_weights = copy.deepcopy(exposeHandler.getWeights())
			class_weights[seen_index_high]-=0.5
			print(class_weights)
			custom_dataloader_train, custom_dataloader_val =\
                exposeHandler.getTrainLoadersOld(seenDataSet, BATCH_SIZE)
            
			model = util.TrainModel(custom_dataloader_train, custom_dataloader_val,\
						len(exposeHandler.getMapping()), model, firstFlag=False, TOTAL_EPOCHS=EPOCHS, class_weights=torch.from_numpy(class_weights).float())


			exposeHandler.updateModel(model)
			exposeHandler.updateDataSet(seenDataSet)
			valLoader = exposeHandler.getValidationSet()
			accuracy_new = util.assessModel(model, valLoader, verbal=False)
			prev_accuracy = accuracy_new



	os.makedirs(PATH, exist_ok=True)

	pickle.dump(exposeHandler.getMapping(),open(os.path.join(PATH, "modelLearnedMapping.pkl"), "wb"))
	torch.save(model, os.path.join(PATH, "final_model.pth"))
	return


def runIncrementalSupervised(classNameList, pickleMap, EXPOSURE_DIR, TEST_DIR, EPOCHS = 15, DIFF_HIGH_THRESHOLD=.5, BATCH_SIZE = 16, FORGET_RATIO=.5, \
	NEW_CLASS_THRES=0.5, DIFF_LOW_THRESHOLD=0.1, DELTA_THRESHOLD = .5, EXEMPLAR_SIZE=300, PATH="mnist_50_imba/", test=False):

	exposureList = np.array(range(len(os.listdir(EXPOSURE_DIR))))
	if test:
		np.random.shuffle(exposureList)
		while (pickleMap[exposureList[0]] == pickleMap[exposureList[1]]):
			np.random.shuffle(exposureList)

	print("Creating Base Exposure")
	exposeHandler, model = createBaseExposures(EXPOSURE_DIR, TEST_DIR, pickleMap, exposureList, BATCH_SIZE, EPOCHS, EXEMPLAR_SIZE)
	valLoader = exposeHandler.getValidationSet()
	prev_accuracy = util.assessModel(model, valLoader, verbal=False)
	print("Base Exposure Accuracy is ")
	print(prev_accuracy)


	EPS = 1e-4

	for i in range(1, len(pickleMap)):
	#for i in range(2,4):
		print("*************************Running Exposure No " + str(i) +":"+ str(exposureList[i]) + " ************************")
		print("New exposure is " + pickleMap[exposureList[i]][0][0])

		previousExposureObj = copy.deepcopy(exposeHandler)
		previous_Model = copy.deepcopy(model)

		dataset = exposeHandler.queryNextExposure(exposureList[i])
		exposeHandler.updateModel(model)
		exposeHandler.updateDataSet(dataset)

		exposeHandler.updateMapping(pickleMap[exposureList[i]])
		class_weights = exposeHandler.getWeights()
        
		exposeHandler.addValidation(len(exposeHandler.getMapping())-1,exposureList[i])
        
        #Assessing the model

		seen_index_high = len(list(exposeHandler.getMapping().keys())) - 1


        # Supervised Labelling
		true_class = pickleMap[exposureList[i]]
		for key in list(exposeHandler.getMapping().keys()):
			if exposeHandler.getMapping()[key] == true_class:
				seen_index_high = key
				break
		if seen_index_high < len(list(exposeHandler.getMapping().keys())) - 1:
			isSeen_high = True
		else:
			isSeen_high = False
            
        # If exposure is a new class
		if isSeen_high == False:
			model = previous_Model
			class_weights = exposeHandler.getWeights()
            
            # Get Train Loaders
			custom_dataloader_train, custom_dataloader_val =\
                exposeHandler.getTrainLoadersNew(BATCH_SIZE)
            
            # Train model with new class
			model = util.TrainModel(custom_dataloader_train, custom_dataloader_val,\
                                        len(exposeHandler.getMapping()), model, firstFlag=False, TOTAL_EPOCHS=EPOCHS, class_weights = torch.from_numpy(class_weights).float())
            
			valLoader = exposeHandler.getValidationSet()
			accuracy_new = util.assessModel(model, valLoader, verbal=False)
            
			prev_accuracy = accuracy_new

        # If it is old class
		else:
			print(classNameList[exposeHandler.getMapping()[seen_index_high][0][1]] + "*******EXPOSURE SEEEENNNNNNNN****")
			model = previous_Model
			exposeHandler = previousExposureObj
			seenDataSet = exposeHandler.queryNextExposure(exposureList[i], assignedLabel=seen_index_high)
			class_weights = copy.deepcopy(exposeHandler.getWeights())
			class_weights[seen_index_high]-=0.5
			custom_dataloader_train, custom_dataloader_val =\
                exposeHandler.getTrainLoadersOld(seenDataSet, BATCH_SIZE)
            
			model = util.TrainModel(custom_dataloader_train, custom_dataloader_val,\
						len(exposeHandler.getMapping()), model, firstFlag=False, TOTAL_EPOCHS=EPOCHS, class_weights=torch.from_numpy(class_weights).float())


			exposeHandler.updateModel(model)
			exposeHandler.updateDataSet(seenDataSet)
			valLoader = exposeHandler.getValidationSet()
			accuracy_new = util.assessModel(model, valLoader, verbal=False)
			prev_accuracy = accuracy_new




	os.makedirs(PATH, exist_ok=True)

	pickle.dump(exposeHandler.getMapping(),open(os.path.join(PATH, "modelLearnedMapping.pkl"), "wb"))
	torch.save(model, os.path.join(PATH, "final_model.pth"))
	return



def createBaseExposures(EXPOSURE_DIR, TEST_DIR, pickleMap,  exposureList, BATCH_SIZE, EPOCHS, EXEMPLAR_SIZE, num_exp=1):

	exposeHandler = ExposureHandler.ExposureHandler(EXPOSURE_DIR, TEST_DIR, exemplar_size = EXEMPLAR_SIZE)
	dataSetList = []

	for i in range(num_exp):
		dataset = exposeHandler.queryNextExposure(exposureList[i])
		exposeHandler.updateMapping(pickleMap[exposureList[i]])
		exposeHandler.updateDataSet(dataset)
		dataSetList.append(dataset)

	concat = ConcatDataset(dataSetList)
	custom_dataloader = DataLoader(concat, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)
	
	


	concat_train = ConcatDataset(exposeHandler.getTrainDataSet())
	concat_val= ConcatDataset(exposeHandler.getValDataSet())
	# exposeHandler.updateMapping(pickleMap[i][1])

	concat_train = TransformedDataset(concat_train, exposeHandler.transform_train)
	concat_val = TransformedDataset(concat_val, exposeHandler.transform_train)
	custom_dataloader_train = DataLoader(concat_train, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)
	custom_dataloader_val = DataLoader(concat_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)

	model = util.TrainModel(custom_dataloader_train, custom_dataloader_val, len(exposeHandler.getMapping()), None, firstFlag=True, TOTAL_EPOCHS=EPOCHS)


	for i in range(num_exp):
		exposeHandler.addValidation(i, exposureList[i])

	return exposeHandler, model



if __name__== "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--cifar10", help="Train on CIFAR10", action="store_true")
	parser.add_argument("--mnist", help="Train on MNIST", action="store_true")
	parser.add_argument("--cifar100", help="Train on CIFAR100", action="store_true")
	parser.add_argument("--svhn", help="Train on SVHN MNIST", action="store_true")
	parser.add_argument("--exposure_size", type=int, help = "Exposure size", default=200)
	parser.add_argument("--exemplar_size", type=int, help = "Per class exemplar size")
	parser.add_argument("--num_exposures", type=int, help = "Number of exposures", default=100)
	parser.add_argument("--epochs", type=int, help = "Epochs per exposure", default=15)
	parser.add_argument("--test", help="Testing flag for shuffling exposures", action='store_true', default=False)
	parser.add_argument("--create_data", help='Create exposures', action='store_true', default=False)
	parser.add_argument("--mode", help='Training mode', default='iLAP_CI')
	parser.add_argument("--path", help='Model save path')
	args = parser.parse_args()
	if not args.exemplar_size:
		args.exemplar_size = args.exposure_size
	if args.cifar10:
		print ("Running CIFAR-10")
		if args.create_data:
			create_exposures('cifar10', args.exposure_size, args.num_exposures)
		with open("exposureMap_mixed_cifar10.pkl", "rb") as input_file:
			pickleMap = pickle.load(input_file)
		EXPOSURE_DIR = "exposures_mixed_cifar10"
		TEST_DIR = "cifar10_data/test/"
		classNameList = ['plane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
		

	elif args.mnist:
		print("Running MNIST")
		classNameList = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
		if args.create_data:
			create_exposures('mnist', args.exposure_size, args.num_exposures)
		with open("exposureMap_mixed_mnist.pkl", "rb") as input_file:
			pickleMap = pickle.load(input_file)
		EXPOSURE_DIR = "exposures_mixed_mnist"
		TEST_DIR = "mnist_data/test/"
		

	elif args.svhn:
		print("Running SVHN")
		classNameList = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
		if args.create_data:
			create_exposures('svhn', args.exposure_size, args.num_exposures)
		with open("exposureMap_mixed_svhn.pkl", "rb") as input_file:
			pickleMap = pickle.load(input_file)
		EXPOSURE_DIR = "exposures_mixed_svhn"
		TEST_DIR = "svhn_data/test/"


	elif args.cifar100:
		print("Running cifar100")
		classNameList = ['streetcar', 'apple', 'palm_tree', 'man', 'forest', 'butterfly', 'lamp', \
			'wardrobe', 'pear', 'raccoon', 'crab', 'tractor', 'aquarium_fish', 'can', 'tank', 'snail', \
			'bed', 'cup', 'television', 'turtle', 'boy', 'mushroom', 'bee', 'fox', 'willow_tree', 'couch',\
	 		'dolphin', 'cattle', 'maple_tree', 'plain', 'bear', 'bridge', 'leopard', 'hamster', 'lawn_mower',\
	  		'lobster', 'camel', 'tiger', 'road', 'whale', 'motorcycle', 'crocodile', 'dinosaur', 'chair',\
	   		'bus', 'plate', 'otter', 'rose', 'seal', 'telephone', 'mouse', 'tulip', 'porcupine', 'beaver',\
	    	'wolf', 'lizard', 'flatfish', 'beetle', 'chimpanzee', 'poppy', 'bowl', 'table', 'shrew', 'skyscraper',\
	     	'kangaroo', 'cloud', 'girl', 'worm', 'train', 'house', 'caterpillar', 'spider', 'rabbit', 'cockroach',\
	      	'rocket', 'castle', 'bicycle', 'baby', 'mountain', 'sweet_pepper', 'orchid', 'sea', 'skunk', \
	      	'oak_tree', 'squirrel', 'ray', 'bottle', 'pickup_truck', 'keyboard', 'lion', 'clock', 'trout', \
	      	'orange', 'woman', 'shark', 'sunflower', 'pine_tree', 'snake', 'elephant', 'possum']
		if args.create_data:
			create_exposures('cifar100', args.exposure_size, args.num_exposures)
		with open("exposureMap_mixed_cifar100.pkl", 'rb') as input_file:
			pickleMap = pickle.load(input_file)
		EXPOSURE_DIR = "exposures_mixed_cifar100"
		TEST_DIR = 'cifar100_data/test/'
		

	else:
		print("DataSet Not Chosen")
		exit()
		
	if args.mode is 'iLAP_CI':
		runIncremental(classNameList, pickleMap, EXPOSURE_DIR, TEST_DIR, EPOCHS=args.epochs, DIFF_HIGH_THRESHOLD=0.4,FORGET_RATIO = 0.5, EXEMPLAR_SIZE =args.exemplar_size, PATH = args.path, test = args.test)
	elif args.mode is 'iLAP_WCI':
		runIncremental(classNameList, pickleMap, EXPOSURE_DIR, TEST_DIR, EPOCHS=args.epochs, DIFF_HIGH_THRESHOLD=0.6,FORGET_RATIO = 1, EXEMPLAR_SIZE =args.exemplar_size, PATH = args.path, test = args.test)
	elif args.mode is 'supervised':
		runIncrementalSupervised(classNameList, pickleMap, EXPOSURE_DIR, TEST_DIR, EPOCHS=args.epochs, DIFF_HIGH_THRESHOLD=0.3,FORGET_RATIO = 0.5, EXEMPLAR_SIZE =args.exemplar_size, PATH = args.path, test = args.test)
	elif args.mode is 'iolfcv':
		runIncrementalIolfcv(classNameList, pickleMap, EXPOSURE_DIR, TEST_DIR, EPOCHS=args.epochs, EXEMPLAR_SIZE =args.exemplar_size, PATH = args.path, test = args.test)
	else:
		print("Illegal training mode")
		exit()
