
from models import ResNet18

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import IncrementalDataSet as IncrementalDataSet
from torchvision import transforms

import numpy as np
import pickle
import argparse
import os



def getTestAccuracy(classNameList, model, test_dir, map_path):
	with open(map_path, 'rb') as input_file:
		label_mapping = pickle.load(input_file)
	mean_color = [0.49131522, 0.48209435, 0.44646862]
	std_color = [0.01897398, 0.03039277, 0.03872553]
	transform = transforms.Compose([
	                 transforms.ToTensor(),
	                 transforms.Normalize(mean_color, std_color),
	            ])

	accuracy = []
	labels_true = np.array([])
	labels_pred = np.array([])
	model.eval()
	inverted_label_map = dict()
	for key, value in label_mapping.items():
		inverted_label_map.setdefault(value[0][1], list()).append(key)
	labelcount = 0
	for className in classNameList:
		test_set = IncrementalDataSet.IncrementalDataSet(labelcount, \
                           os.path.join(test_dir, className),transform)
		test_loader = DataLoader(test_set, batch_size=64, drop_last=False, shuffle=True)

		total = 0
		count = 0

		for batch_idx, (data, target) in enumerate(test_loader):
			if torch.cuda.is_available():
				data, target = data.cuda(), target.cuda()
			y_pred = model(data)

			pred = np.argmax(y_pred.cpu().detach().numpy(),axis=1)


			ground_true = target.cpu().detach().numpy()

			if labelcount in inverted_label_map.keys():
				for i in range(len(pred)):
					if pred[i] in inverted_label_map[ground_true[i]]:
						count += 1/len(inverted_label_map[ground_true[i]])
					total += 1
			labels_true = np.concatenate((labels_true, ground_true))
			labels_pred = np.concatenate((labels_pred, pred))

		if labelcount not in inverted_label_map.keys():
			labelcount+=1
			continue

		accuracy.append(count/total)
		labelcount += 1

	accuracyPerClass = {}

	for i in range(len(accuracy)):
		accuracyPerClass[classNameList[i]] = accuracy[i]

	return accuracyPerClass, np.sum(accuracy)/len(classNameList)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--cifar10", help="Assess CIFAR-10", action="store_true")
	parser.add_argument("--mnist", help="Assess MNIST", action="store_true")
	parser.add_argument("--cifar100", help="Assess CIFAR-100", action="store_true")
	parser.add_argument("--svhn", help="Assess SVHN-MNIST", action="store_true")
	parser.add_argument("--model", type=str, help = "Evaluation Model")
	parser.add_argument("--test_dir", help="Test Directory")
	parser.add_argument("--no_classes", type=int, help = "Number of Classes in DataSet")
	parser.add_argument("--map_path", type=str, help = "Path to Ground Truth Mapping")

	args = parser.parse_args()
	if args.cifar10:
		classNameList = ['plane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	elif args.cifar100:
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
	elif args.mnist:
		classNameList = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
	elif args.svhn:
		classNameList = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

	model = torch.load(args.model)
	accuracyPerClass, accuracy = getTestAccuracy(classNameList, model, args.test_dir, args.map_path)
	print(accuracyPerClass)
	print("Total Accuracy: " + str(accuracy))
