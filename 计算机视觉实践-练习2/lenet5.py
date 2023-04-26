# import packages
import torch
import torchvision
# Device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
num_epochs = 10
num_classes = 10
batch_size = 100
learning_rate = 0.001
momentum = 0.9
# Load downloaded dataset.
import numpy as np
import gzip
import os
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.file_pre = 'train' if train == True else 't10k'
        self.transform = transform
        self.label_path = os.path.join(root, '%s-labels-idx1-ubyte.gz' % self.file_pre)
        self.image_path = os.path.join(root, '%s-images-idx3-ubyte.gz' % self.file_pre)
        self.images, self.labels = self.__read_data__(self.image_path, self.label_path)
    
    def __read_data__(self, image_path, label_path):
        # Read dataset.
        with gzip.open(label_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(image_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(labels), 28, 28)
        return images, labels
    
    def __getitem__(self, index):
        image, label = self.images[index], int(self.labels[index])
        if self.transform is not None:
            image = self.transform(np.array(image))
        return image, label
    
    def __len__(self):
        return len(self.labels)
    
train_dataset = MNISTDataset('C:/Users/80712/Desktop/shiyan2/data/MNIST/', transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1037,), (0.3081,))]))
test_dataset = MNISTDataset('C:/Users/80712/Desktop/shiyan2/data/MNIST/', train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1037,), (0.3081,))]))
# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def fc_in(image, Conv, Pool):
    for i, j in zip(Conv, Pool):
        hk = (image[0] - i[0] + 2 * i[2]) / i[1] + 1
        wk = (image[1] - i[0] + 2 * i[2]) / i[1] + 1
        hp = (hk - j[0] + 2 * j[2]) / j[1] + 1
        wp = (wk - j[0] + 2 * j[2]) / j[1] + 1
        image = (hp, wp)
    return (int(image[0]), int(image[1]))
fc_in((28, 28), ((5, 1, 0), (5, 1, 0)), ((2, 2, 0), (2, 2, 0)))

# LeNet-5
class LeNet5(torch.nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                                          torch.nn.BatchNorm2d(6),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(4 * 4 * 16, 120),
                                       torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(120, 84),
                                       torch.nn.ReLU())
        self.fc3 = torch.nn.Linear(84, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# Make model
model = LeNet5(num_classes).to(device)
# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item())) 

# Test the model.
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print ('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model.
torch.save(model.state_dict(), 'LeNet5.ckpt')
