import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.functional import softmax, log_softmax, kl_div, sigmoid
import torch.optim as optim
import itertools
from torch.utils.data import TensorDataset
from TransformedDataset import TransformedDataset
from torch.autograd import Variable
from models import ResNet18
import models
import pdb
import copy


def TrainModel(train_loader, val_loader, output_length, prev_model, firstFlag=False, TOTAL_EPOCHS=15, class_weights=None):
	losses = []
	# Changing final layer
	if firstFlag:
		model = ResNet18(output_length)
	else:
		"Changing Layers"
		model = prev_model
		prev_model.changeFC3(output_length)

	if torch.cuda.is_available():
		model.cuda()

	lr = 2e-4

	optimizer = optim.Adam([
		{
        "params": model.classifier.parameters(),
        "lr": lr * 0.1,
    },
		{
        "params": model.fc.parameters(), 
        "lr": lr
    }
    ])
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model.train()
	best_model = None
	best_loss = float("inf")


	for epoch in range(TOTAL_EPOCHS):
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
#         # Get Samples
			
			if torch.cuda.is_available():
				data, target = Variable(data.cuda()), Variable(target.cuda())
				if class_weights is not None:
					class_weights = Variable(class_weights.cuda())
			optimizer.zero_grad()

			y_pred = model(data)
			if class_weights is not None:
				loss = F.cross_entropy(y_pred, target, weight=class_weights)
			else:
				loss = F.cross_entropy(y_pred, target)

			losses.append(loss.item())
			loss.backward()
			optimizer.step()

			if batch_idx % 100 == 1:

				print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						epoch,
						epoch * len(data),
						TOTAL_EPOCHS * len(data),
						100. * ((epoch * len(data))/(TOTAL_EPOCHS * len(data))),
						loss.item()),
						end='')
			del loss, data, target, y_pred
		loss = 0
		model.eval()
		for batch_idx, (data, target) in enumerate(val_loader):
			# Get Samples
			if torch.cuda.is_available():
				data, target = Variable(data.cuda()), Variable(target.cuda())
			optimizer.zero_grad()
			y_pred = model(data)

			if class_weights is not None:
				loss += F.cross_entropy(y_pred, target, weight=class_weights)
			else:
				loss += F.cross_entropy(y_pred, target)

		print('\r Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch,
				epoch * len(data),
				TOTAL_EPOCHS * len(data),
				100. * ((epoch * len(data))/(TOTAL_EPOCHS * len(data))),
				loss.item()),
				end='')

		
		# Save best model
		if loss < best_loss:
			best_model = copy.deepcopy(model)
			best_loss = loss
		del loss, data, target, y_pred

	print("\n")

	model = best_model
	optimizer = optim.Adam([
		{
        "params": [model.alpha, model.beta],
        "lr": lr,
    }
    ])
	for epoch in range(TOTAL_EPOCHS):
		model.train()
		for batch_idx, (data, target) in enumerate(val_loader):
			
			if torch.cuda.is_available():
				data, target = Variable(data.cuda()), Variable(target.cuda())
				if class_weights is not None:
					class_weights = Variable(class_weights.cuda())
			optimizer.zero_grad()
			y_pred = model(data)
			
			if class_weights is not None:
				loss = F.cross_entropy(y_pred, target, weight=class_weights)
			else:
				loss = F.cross_entropy(y_pred, target)
			losses.append(loss.item())
			loss.backward()
			optimizer.step()
			#scheduler.step()
			if batch_idx % 100 == 1:

				print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						epoch,
						epoch * len(data),
						TOTAL_EPOCHS * len(data),
						100. * ((epoch * len(data))/(TOTAL_EPOCHS * len(data))),
						loss.item()),
						end='')
			del loss, data, target, y_pred
	return model


def sample_best(seenDataSet, model, exposeHandler, num_samples = 300, reduce=False):
	if num_samples >= len(seenDataSet):
		return seenDataSet
	if reduce:
		return [seenDataSet[i] for i in range(num_samples)]
	epsilon = 1e-16
    
	concat_val = TransformedDataset(seenDataSet, exposeHandler.transform)
	val_loader = DataLoader(concat_val, batch_size=1, drop_last=False, shuffle=False)
	model_copy = copy.deepcopy(model)
	model_copy.eval()
	model_copy.fc = None
	predictions = []
	indices = []
	if torch.cuda.is_available():
		model_copy.cuda()
	
	# Get exemplar distances
	for batch_idx, (data,target) in enumerate(val_loader):
		if torch.cuda.is_available():
			data, target = Variable(data.cuda()), Variable(target.cuda())
		features = model_copy(data)
		features_norm = features.data.norm(p=2, dim=1) + epsilon
		features_norm = features_norm.unsqueeze(1)
		features.data = features.data.div(
                features_norm.expand_as(features))  # Normalize
		pred = features.data.cpu().numpy()[0]
		predictions.append(pred)
	mean_pred = np.mean(predictions, axis=0)
	predictions_chosen = np.zeros(pred.shape[0])

	# Get best samples
	num_samples = int(num_samples)
	for i in range(num_samples):
		index_sorted = np.argsort(np.linalg.norm(mean_pred - (predictions_chosen + np.array(predictions))/(i+1), axis=1))
		for idx in index_sorted:
			if idx not in indices:
				index = idx
				break
		predictions_chosen += predictions[index]
		indices.append(index)
	best_dataset = [seenDataSet[i] for i in indices]
	return best_dataset
	
def assessModel(model, tests, verbal=False):
	accuracy = []
	model.eval()
	if torch.cuda.is_available():
		model.cuda()
	for test_set in tests:
		count = 0
		total = 0
		first = True
		for batch_idx, (data, target) in enumerate(test_set):
			if torch.cuda.is_available():
				data, target = Variable(data.cuda()), Variable(target.cuda())
			y_pred = model(data)

			pred = np.argmax(y_pred.cpu().detach().numpy(),axis=1)

			ground_true = target.cpu().detach().numpy()

			if verbal:
				print("\n")
				print("Assessing Target " + str(ground_true[0]))
				print("Actual Image " + str(self.dict[ground_true[0]]))


			for i in range(len(pred)):
				if pred[i] == ground_true[i]:
					count +=1
				total += 1

		accuracy.append(count/total)

	return accuracy



def classIsSeen(prev_acc, curr_acc, THRESHOLD):

	maxLength = min(len(prev_acc), len(curr_acc))
	highestDiff = 100
	index = -1

	for i in range(maxLength):
		if prev_acc[i] == 0:
			highestDiff = curr_acc[i]/.001
			index = i

		elif curr_acc[i]/prev_acc[i] < THRESHOLD and curr_acc[i]/prev_acc[i] < highestDiff:
			highestDiff = curr_acc[i]/prev_acc[i]
			index = i

	if index != -1:
		return True, index

	return False, None

def getDistances(model, test_set, verbal=False):
	mean_distances = []
	model_copy = copy.deepcopy(model)
	model_copy.eval()
	model_copy.fc = None
	if torch.cuda.is_available():
		model_copy.cuda()
	epsilon = 1e-16
	
	count = 0
	total = 0
	first = True
	features = None
	for batch_idx, (data, target) in enumerate(test_set):
		if torch.cuda.is_available():
			data, target = Variable(data.cuda()), Variable(target.cuda())
		feature = model_copy(data)

		feature_norm = feature.data.norm(p=2, dim=1) + epsilon
		feature_norm = feature_norm.unsqueeze(1)
		feature.data = feature.data.div(feature_norm.expand_as(feature))
		if features is None:
			features = feature.cpu().detach().numpy()
		else:
			features = np.concatenate((features, feature.cpu().detach().numpy()),axis=0)

	print(features.shape)


	return features

