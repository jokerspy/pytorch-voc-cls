import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import PIL
from data import VOCData
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import numpy as np
import os
from tqdm import tqdm

USER_DIR = '/users/im64/'
VOC_DIR = USER_DIR + 'VOC_yolo'
post_fix = 'experiment_3'
try:
	os.mkdir('snapshot/%s'%post_fix)
except:
	pass
writer = SummaryWriter('runs/%s'%post_fix)
torch.manual_seed(2019)
np.random.seed(2019)
if torch.cuda.is_available():
	print('Use GPU')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#alex_net = AlexNet().to(device)
alex_net = models.alexnet(pretrained=True)
for param in alex_net.features.parameters():
	param.requires_grad = False
alex_net.classifier[-1] = nn.Linear(4096, 20)
alex_net.to(device)
print(list(alex_net.classifier[-1].parameters())[:])
print(alex_net)

mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

train_transform = transforms.Compose([transforms.Resize((227, 227)),
#                                      transforms.RandomChoice([
#                                              transforms.CenterCrop(300),
#                                              transforms.RandomResizedCrop(300, scale=(0.80, 1.0)),
#                                              ]),                                      
                                      transforms.RandomChoice([
                                          transforms.ColorJitter(brightness=(0.80, 1.20)),
                                          transforms.RandomGrayscale(p = 0.25)
                                          ]),
                                      transforms.RandomHorizontalFlip(p = 0.25),
                                      transforms.RandomRotation(25),
                  transforms.ToTensor(),
                  transforms.Normalize(mean, std)])

valid_transform = transforms.Compose(
                 [transforms.Resize(330), 
                                          transforms.CenterCrop(227), 
                  transforms.ToTensor(),
                  #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                  transforms.Normalize(mean, std)])

trainset = VOCData(root=VOC_DIR, year='2007',
                   image_set='train', download=False, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

validset = VOCData(root=VOC_DIR, year='2007',
                   image_set='val', download=False, transform=valid_transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=16,
                                          shuffle=True, num_workers=2)

def get_mAP(gts, scr):
	aps = []
	for i in range(gts.shape[0]):
		# Subtract eps from score to make AP work for tied scores
		try:
			ap = metrics.average_precision_score(gts[i][gts[i] != .5], scr[i][gts[i] != .5]-1e-5*gts[i][gts[i] != .5])
		except:
			print(gts[i], scr[i])
		aps.append( ap )
	print( np.mean(aps), '  ', ' '.join(['%0.2f'%a for a in aps]) )
	return np.mean(aps), aps

criterion = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.SGD(alex_net.classifier.parameters(), lr=1.5e-4, momentum=0.9, weight_decay=5e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, cooldown=2)
#optimizer = optim.SGD([   
#        {'params': list(alex_net.parameters())[:-1], 'lr': 1.5e-4, 'momentum': 0.9},
#        {'params': list(alex_net.parameters())[-1], 'lr': .01, 'momentum': 0.9}
#        ])
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)
for epoch in range(100):  # loop over the dataset multiple times
	print('Training epoch %d' % (epoch + 1))
	train_loss = 0.0
	valid_loss = 0.0
	n_img = 0
	alex_net.train()
	if epoch == 20:
		for param in alex_net.features.parameters():
			param.requires_grad = True
		optimizer = optim.SGD(alex_net.parameters(), lr=1.5e-5, momentum=0.9, weight_decay=5e-7)
	cnt = sum(p.numel() for p in alex_net.parameters() if p.requires_grad)
	print('Learnable params: %d'%cnt)
	gts, scr = np.ndarray([0, len(VOCData.categories)]), np.ndarray([0, len(VOCData.categories)])
	for i, data in tqdm(enumerate(trainloader)):
		inputs, labels = data
		writer.add_graph(alex_net, inputs.to(device))

		optimizer.zero_grad()
		outputs = alex_net(inputs.to(device))
		loss = criterion(outputs, labels.to(device))


		gts = np.concatenate((gts, labels.cpu().detach().numpy()))
		scr = np.concatenate((scr, outputs.cpu().detach().numpy()))
		train_loss += loss.item()
		n_img += len(labels)
		if i % 10 == 9:
			print('[%d, %5d, %5d] loss: %.3f' % (epoch + 1, i + 1, n_img, loss.item()/len(labels)))
		loss.backward()
		optimizer.step()

		del inputs, labels, outputs
		torch.cuda.empty_cache()
	train_loss /= len(trainset)
	mAP, _ = get_mAP(gts.T, scr.T)
	writer.add_scalar('Loss/Train', train_loss, epoch)
	writer.add_scalar('Acc/Train', mAP, epoch)
	if epoch % 20 == 19:
		torch.save({
		           'epoch': epoch,
		           'model_state_dict': alex_net.state_dict(),
		           'optimizer_state_dict': optimizer.state_dict(),
		           'loss': loss,
		           }, 'snapshot/%s/alexnet_%d'%(post_fix, epoch))
	print('Train epoch %d loss: %.3f' % (epoch + 1, train_loss))
	alex_net.eval()
	gts, scr = np.ndarray([0, len(VOCData.categories)]), np.ndarray([0, len(VOCData.categories)])
	for i, data in tqdm(enumerate(validloader)):
		inputs, labels = data
		outputs = alex_net(inputs.to(device))
		loss = criterion(outputs, labels.to(device))
		valid_loss += loss.item()
		gts = np.concatenate((gts, labels.cpu().detach().numpy()))
		scr = np.concatenate((scr, outputs.cpu().detach().numpy()))
	valid_loss /= len(validset)
	print('Valid epoch %d loss: %.3f' % (epoch + 1, valid_loss))
	mAP, _ = get_mAP(gts.T, scr.T)
	#scheduler.step(valid_loss)
	#scheduler.step()
	writer.add_scalar('Loss/Valid', valid_loss, epoch)
	writer.add_scalar('Acc/Valid', mAP, epoch)

writer.close()
torch.save({
           'epoch': epoch,
           'model_state_dict': alex_net.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           'loss': loss,
           }, 'snapshot/%s/alexnet'%post_fix)
'''checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
model.train()'''
print('Finished Training')