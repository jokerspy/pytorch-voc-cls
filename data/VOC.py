import torch
import torchvision.datasets

class VOCData(torchvision.datasets.VOCDetection):
	num_categories = 20
	categories = ['aeroplane', 'bicycle', 'bird', 'boat',
	              'bottle', 'bus', 'car', 'cat', 'chair',
	              'cow', 'diningtable', 'dog', 'horse',
	              'motorbike', 'person', 'pottedplant',
	              'sheep', 'sofa', 'train', 'tvmonitor']

	def __getitem__(self, index):
		img, meta = super().__getitem__(index)
		labels = torch.zeros([len(self.categories)])
		obj_list = meta['annotation']['object']
		if isinstance(obj_list, list):
			for obj in obj_list:
				idx = self.categories.index(obj['name'])
				if int(obj['difficult']):
					labels[idx] = .5
				else:
					labels[idx] = 1
		else:
			idx = self.categories.index(obj_list['name'])
			if int(obj_list['difficult']):
				labels[idx] = 255
			else:
				labels[idx] = 1
		return img, labels

	def __len__(self):
		return len(self.images)
