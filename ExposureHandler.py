import os
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import IncrementalDataSet as IncrementalDataSet
from TransformedDataset import TransformedDataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import util
from RandAugment import RandAugment
import numpy as np
import pdb
import copy
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

class ExposureHandler(object):
	def __init__(self, EXPOSURE_DIR, TEST_DIR, exemplar_size = 100, val_ratio=0.8, size=(224,224)):
		self.label_mapping = {}
		self.exposure_dir = EXPOSURE_DIR
		self.test_dir = TEST_DIR
		self.train_dataset = []
		self.class_weights = np.array([])
		self.val_dataset = []
		#Array of validators
		self.validationLoader = []
		self.val_ratio = val_ratio
		cifar10_mean_color = [0.49131522, 0.48209435, 0.44646862]
		# std dev of color across training images
		cifar10_std_color = [0.01897398, 0.03039277, 0.03872553]
		self.transform = transforms.Compose([
						transforms.Resize(size),
		                 transforms.ToTensor(),
		                 transforms.Normalize(cifar10_mean_color, cifar10_std_color),
		            ])
		self.transform_train = transforms.Compose([
						transforms.Resize(size),
		                 transforms.RandomHorizontalFlip(),
		                 transforms.RandomCrop(size=size, padding=4),
		                 transforms.ToTensor(),
		                 transforms.Normalize(cifar10_mean_color, cifar10_std_color),
		            ])
		self.random_seed = 7
		# Add RandAugment with N, M(hyperparameter)
		N = 1
		M = 2
		self.model = None
		self.transform_train.transforms.insert(0, RandAugment(N, M))
		self.exemplar_size = exemplar_size
	
	def updateModel(self, model):
		self.model = model

	def queryNextExposure(self, ExposureNo, assignedLabel=None):
		dataset = None
		if assignedLabel == None:
			dataset = IncrementalDataSet.\
			IncrementalDataSet(len(self.label_mapping), os.path.join(self.exposure_dir, str(ExposureNo)),None)
		else:
			dataset = IncrementalDataSet.\
			IncrementalDataSet(assignedLabel, os.path.join(self.exposure_dir, str(ExposureNo)), None)

		return dataset

	def addValidation(self, assignedLabel, ExposureNo, BATCH_SIZE=16):
		dataset = IncrementalDataSet.IncrementalDataSet(assignedLabel, \
                                                        os.path.join(self.exposure_dir, str(ExposureNo)),self.transform)
		dataset_size = len(dataset)
		indices = list(range(dataset_size))
		split = int(np.floor(self.val_ratio * dataset_size))
		np.random.seed(self.random_seed)
		np.random.shuffle(indices)

		train_indices, val_indices = indices[:split], indices[split:]

		# Creating PT data samplers and loaders:
		train_sampler = SubsetRandomSampler(train_indices)
		valid_sampler = SubsetRandomSampler(val_indices)
		loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=False, sampler=valid_sampler)
		self.validationLoader.append(loader)
		
	
	def getValidation(self, assignedLabel, dataset_in, BATCH_SIZE):
		dataset = [(dataset_in[i][0], assignedLabel) for i in range(len(dataset_in))]
		dataset_val = TransformedDataset(dataset, self.transform)
		loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)
		return loader


	def updateDataSet(self, dataset):
		max_samples = 500
		dataset_size = len(dataset)
		indices = list(range(dataset_size))
		split = int(np.floor(self.val_ratio * dataset_size))
		np.random.seed(self.random_seed)
		np.random.shuffle(indices)
		train_indices, val_indices = indices[:split], indices[split:]

		dataset_label = dataset[0][1]
		exemplar_split = int(np.floor(self.exemplar_size*self.val_ratio))
		if np.size(self.class_weights) == dataset_label:
			self.class_weights = np.append(self.class_weights,[1])
			self.train_dataset.append(util.sample_best([dataset[i] for i in train_indices], self.model, self,\
													   exemplar_split))
			self.val_dataset.append(util.sample_best([dataset[i] for i in val_indices], self.model, self, \
													 self.exemplar_size -  exemplar_split))
			
		else:
			self.train_dataset[dataset_label] = util.sample_best(ConcatDataset([self.train_dataset[dataset_label],[dataset[i] for i in train_indices]]), self.model, self, exemplar_split)
			self.val_dataset[dataset_label] = util.sample_best(ConcatDataset([self.val_dataset[dataset_label],[dataset[i] for i in val_indices]]), self.model, self, self.exemplar_size -  exemplar_split)

	def getWeights(self):
		return np.max(self.class_weights)*np.reciprocal(self.class_weights)

	def getFeedbackLoaders(self, FORGET_RATIO, BATCH_SIZE):
		concat_train = ConcatDataset([ data[:int(len(data)*FORGET_RATIO)] for data in self.getTrainDataSet()[:-1]]+[self.getTrainDataSet()[-1]])
		concat_train = TransformedDataset(concat_train, self.transform_train)
		concat_val = ConcatDataset([ data[:int(len(data)*FORGET_RATIO)] for data in self.getValDataSet()[:-1]]+[self.getValDataSet()[-1]])
		concat_val = TransformedDataset(concat_val, self.transform)
		custom_dataloader_train = DataLoader(concat_train, batch_size=BATCH_SIZE, num_workers=4, drop_last=False, shuffle=True)
		custom_dataloader_val = DataLoader(concat_val, batch_size=BATCH_SIZE, num_workers=4, drop_last=False, shuffle=False)
		return custom_dataloader_train, custom_dataloader_val
    
	def getTrainLoadersNew(self, BATCH_SIZE):
		concat_train = ConcatDataset(self.getTrainDataSet())
		concat_val= ConcatDataset(self.getValDataSet())
		
		concat_train = TransformedDataset(concat_train, self.transform_train)
		concat_val = TransformedDataset(concat_val, self.transform)
		
		custom_dataloader_train = DataLoader(concat_train, batch_size=BATCH_SIZE, num_workers=4, drop_last=False, shuffle=True)
		custom_dataloader_val = DataLoader(concat_val, batch_size=BATCH_SIZE, num_workers=4, drop_last=False, shuffle=False)
		return custom_dataloader_train, custom_dataloader_val
	
	def getDistanceLoaders(self, dataset, BATCH_SIZE):
		custom_dataloaders_exemplars = []
		for dataset_val in self.getTrainDataSet():
			dataset_val_trans = TransformedDataset(dataset_val, self.transform)
			dataloader_val = DataLoader(dataset_val_trans, batch_size=BATCH_SIZE, num_workers=4, drop_last=False, shuffle=False)
			custom_dataloaders_exemplars.append(dataloader_val)
		if dataset is not None:
			dataset_val_trans = TransformedDataset(dataset, self.transform)
			dataloader_val = DataLoader(dataset_val_trans, batch_size=BATCH_SIZE, num_workers=4, drop_last=False, shuffle=False)
		else:
			dataloader_val = None
		return custom_dataloaders_exemplars, dataloader_val
    
	def getTrainLoadersOld(self, seenDataSet, BATCH_SIZE):
		dataset_size = len(seenDataSet)
		indices = list(range(dataset_size))
		split = int(np.floor(self.val_ratio * dataset_size))
		np.random.seed(self.random_seed)
		np.random.shuffle(indices)
		train_indices, val_indices = indices[:split], indices[split:]
		previousExemplars_train = copy.deepcopy(self.getTrainDataSet())
		previousExemplars_val = copy.deepcopy(self.getValDataSet())
		previousExemplars_train.append([seenDataSet[i] for i in train_indices])
		previousExemplars_val.append([seenDataSet[i] for i in val_indices])
		concat_train = ConcatDataset(previousExemplars_train)
		concat_val = ConcatDataset(previousExemplars_val)
		concat_train = TransformedDataset(concat_train, self.transform_train)
		concat_val = TransformedDataset(concat_val, self.transform)

		custom_dataloader_train = DataLoader(concat_train, batch_size=BATCH_SIZE, num_workers=4, drop_last=False, shuffle=True)
		custom_dataloader_val = DataLoader(concat_val, batch_size=BATCH_SIZE, num_workers=4, drop_last=False, shuffle=False)
		return custom_dataloader_train, custom_dataloader_val
    
	def getFinalValidationAccuracy(self, classNameList, model):
		accuracy = []
		labels_true = np.array([])
		labels_pred = np.array([])
		model.eval()
		inverted_label_map = dict()
		for key, value in self.label_mapping.items():
			inverted_label_map.setdefault(value[0][1], list()).append(key)
		labelcount = 0
		for className in classNameList:
			test_set = IncrementalDataSet.IncrementalDataSet(labelcount, \
                            os.path.join(self.test_dir, className),self.transform)
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
							count +=1
						total += 1
				labels_true = np.concatenate((labels_true, ground_true))
				labels_pred = np.concatenate((labels_pred, pred))
			
			if labelcount not in inverted_label_map.keys():
				labelcount+=1
				continue

			accuracy.append(count/total)
			labelcount += 1
		ari = adjusted_rand_score(labels_true, labels_pred)
		nmi = normalized_mutual_info_score(labels_true, labels_pred)

		return accuracy, ari, nmi


	def getTrainDataSet(self):
		return self.train_dataset

	def getValDataSet(self):
		return self.val_dataset

	def getValidationSet(self):
		return self.validationLoader

	def getMapping(self):
		return self.label_mapping

	#Assigned Index : Actual Class Label
	def updateMapping(self, label):
		self.label_mapping[len(self.label_mapping)] = label

	def overrideMapping(self, mapping):
		self.label_mapping = mapping

	def getTrainIndexCount(self):
		return self.train_index_count
	
	def removeClass(self, index, BATCH_SIZE):
		self.label_mapping[index] = copy.deepcopy(self.label_mapping[len(self.label_mapping)-1])
		self.label_mapping.pop(len(self.label_mapping)-1)
		train_dataset = copy.deepcopy(self.train_dataset[-1])
		self.train_dataset[index] = [(train_dataset[i][0],index) for i in range(len(train_dataset))]
		self.train_dataset.pop()
		val_dataset = copy.deepcopy(self.val_dataset[-1])
		self.val_dataset[index] = [(val_dataset[i][0],index) for i in range(len(val_dataset))]
		self.validationLoader[index] = self.getValidation(index, self.val_dataset[index], BATCH_SIZE)
		self.val_dataset.pop()
		self.validationLoader.pop()
		self.class_weights[index] = copy.deepcopy(self.class_weights[-1])
		self.class_weights = self.class_weights[:-1]
