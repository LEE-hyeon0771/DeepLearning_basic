import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# hyper parameters
batch_size = 128
lr = 0.005
epochs = 30

# load data
train_dataset = datasets.MNIST(root="./data/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="./data/", train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# define network for Case 1
class Net1(nn.Module):
  def __init__(self):
    super(Net1, self).__init__()
    self.fc1 = nn.Linear(784, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 16)
    self.fc4 = nn.Linear(16, 10)

  def forward(self, x):
    x = x.float()
    h1 = torch.tanh(self.fc1(x.view(-1, 784)))
    h2 = torch.tanh(self.fc2(h1))
    h3 = torch.tanh(self.fc3(h2))
    h4 = self.fc4(h3)
    return F.log_softmax(h4, dim=1)

# define network for Case 2
class Net2(nn.Module):
  def __init__(self):
    super(Net2, self).__init__()
    self.fc1 = nn.Linear(784, 16)
    self.fc2 = nn.Linear(16, 10)

  def forward(self, x):
    x = x.float()
    h1 = torch.tanh(self.fc1(x.view(-1, 784)))
    h2 = self.fc2(h1)
    return F.log_softmax(h2, dim=1)

model1 = Net1()
model2 = Net2()
optimizer1 = optim.Adam(model1.parameters(), lr=lr)
optimizer2 = optim.Adam(model2.parameters(), lr=lr)

# Define train and test function
def train(model, optimizer, epoch):
  model.train()
  train_loss=[]
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 100 == 0:
      print("Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}".format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    train_loss.append(loss.item())
  return sum(train_loss)/len(train_loss)

def test(model):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = Variable(data), Variable(target)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).cpu().sum()
  test_loss /= len(test_loader.dataset)
  print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
  return test_loss

def accuracy(model, loader):
  model.eval()
  correct = 0
  with torch.no_grad():
    for data, target in loader:
      output = model(data)
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()
  return 100. * correct / len(loader.dataset)

# Train for Case 1
train_loss1=list()
test_loss1=list()
for epoch in range(1, epochs+1):
  train_loss1.append(train(model1, optimizer1, epoch))
  test_loss1.append(test(model1))

# Train for Case 2
train_loss2=list()
test_loss2=list()
for epoch in range(1, epochs+1):
  train_loss2.append(train(model2, optimizer2, epoch))
  test_loss2.append(test(model2))

# Show loss graph
plt.figure()
plt.title("Loss graph for Case 1")
plt.plot(train_loss1, marker='.', c='blue', label="Train-set Loss")
plt.plot(test_loss1, marker='.', c='red', label="Validation-set Loss")
plt.legend()
plt.show()

plt.figure()
plt.title("Loss graph for Case 2")
plt.plot(train_loss2, marker='.', c='blue', label="Train-set Loss")
plt.plot(test_loss2, marker='.', c='red', label="Validation-set Loss")
plt.legend()
plt.show()

# Show prediction result for model1
images, labels = next(iter(train_loader))
output = model1(images[0])
pred = output.data.max(1, keepdim=True)[1]

plt.imshow(images[0].reshape(28,28), cmap="gray")
plt.title(f"Predicted: {pred.item()}, Actual: {labels[0].item()}")
plt.show()

# Show prediction result for model2
images, labels = next(iter(train_loader))
output = model2(images[0])
pred = output.data.max(1, keepdim=True)[1]

plt.imshow(images[0].reshape(28,28), cmap="gray")
plt.title(f"Predicted: {pred.item()}, Actual: {labels[0].item()}")
plt.show()

print("Accuracy of model 1: {:.2f}%".format(accuracy(model1, test_loader)))
print("Accuracy of model 2: {:.2f}%".format(accuracy(model2, test_loader)))