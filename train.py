import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import PIL
from data import VOCData

if torch.cuda.is_available():
	print('Use GPU')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

alex_net = AlexNet().to(device)
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

trainset = VOCData(root=VOC_DIR, year='2007',
                   image_set='train', download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = VOCData(root=VOC_DIR, year='2007',
                  image_set='test', download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=True, num_workers=2)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(alex_net.parameters(), lr=0.01, momentum=0.9, weight_decay=.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, cooldown=2)

for epoch in range(200):  # loop over the dataset multiple times
	print('Training epoch %d' % (epoch + 1))
	train_loss = 0.0
	valid_loss = 0.0
	#accum_loss = 0.0
	n_img = 0
	for i, data in enumerate(trainloader):
		inputs, labels = data

		optimizer.zero_grad()

		outputs = alex_net(inputs.to(device))

		loss = criterion(outputs.to(device), labels.to(device))
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		#accum_loss += loss.item() * len(labels)
		n_img += len(labels)
		if i % 10 == 9:
			print('[%d, %5d, %5d] loss: %.3f' % (epoch + 1, i + 1, n_img, train_loss))
		train_loss = 0
	#print('Train epoch %d loss: %.3f' % (epoch + 1, accum_loss / len(trainset)))

	for i, data in enumerate(testloader):
		inputs, labels = data
		outputs = alex_net(inputs.to(device))
		loss = criterion(outputs.to(device), labels.to(device))
		valid_loss += loss.item() * len(labels)
	valid_loss /= len(testset)
	print('Valid epoch %d loss: %.3f' % (epoch + 1, valid_loss))
	scheduler.step(valid_loss)

print('Finished Training')