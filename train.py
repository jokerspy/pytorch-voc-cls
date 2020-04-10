import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import PIL

alex_net = AlexNet()
print(alex_net)

USER_DIR = '/users/im64/'
VOC_DIR = USER_DIR + 'VOC_yolo'

input = torch.randn(1, 1, 227, 227)

transform = transforms.Compose(
    [transforms.RandomResizedCrop((227, 227), scale=(0.8, 1.2)),
     transforms.RandomVerticalFlip(),
     transforms.RandomHorizontalFlip(),
     transforms.ColorJitter(),
     transforms.RandomRotation(15., resample=PIL.Image.BILINEAR),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.VOCDetection(root=VOC_DIR, year='2007',
                                        image_set='train', download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=2)

print(trainset, trainloader)
testset = torchvision.datasets.VOCDetection(root=VOC_DIR, year='2007',
                                        image_set='test', download=False, transform=None)
testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alex_net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(trainloader):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data
		print(labels)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = alex_net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:	# print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')