import torch
import torchvision.datasets
import numpy as np

class VOCData(torchvision.datasets.VOCDetection):
	categories = ['aeroplane', 'bicycle', 'bird', 'boat',
	              'bottle', 'bus', 'car', 'cat', 'chair',
	              'cow', 'diningtable', 'dog', 'horse',
	              'motorbike', 'person', 'pottedplant',
	              'sheep', 'sofa', 'train', 'tvmonitor']

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		image, target = super().__getitem__(index)
		obj_list = target['annotation']['object']

		pos_idx = []
		diff_idx = []
		if type(obj_list) == dict:
			if int(obj_list['difficult']) == 0:
				pos_idx.append(self.categories.index(obj_list['name']))
			else:
				diff_idx.append(self.categories.index(obj_list['name']))

		else:
			for i in range(len(obj_list)):
				if int(obj_list[i]['difficult']) == 0:
					pos_idx.append(self.categories.index(obj_list[i]['name']))
				else:
					diff_idx.append(self.categories.index(obj_list[i]['name']))

		labels = np.zeros(len(self.categories))
		labels[pos_idx] = 1
		#labels[diff_idx] = .5

		return image, torch.from_numpy(labels)