import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# Loading data
train_data = torchvision.datasets.CIFAR10(root='../data',
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root='../data',
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# Checking datasize
train_data_size = len(train_data)
test_data_size = len(test_data)

print('Training data size: {}'.format(train_data_size))
print('Testing data size: {}'.format(test_data_size))

# Batch loading using dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Build model
model = ModelClassifier()

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
lr = 1e-2
opt = torch.optim.SGD(model.parameters(), lr=lr)

# Add tensorboard to visualize training
writer = SummaryWriter('../logs_train')

# Training params
total_steps = 0
epoch = 1000

for i in range(epoch):
    print('Training step {}'.format(i))
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        output = model(imgs)
        loss = loss_fn(output, targets)

        # optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_steps += 1
        if total_steps % 100 == 0:
            print('Total step {} Loss {}'.format(total_steps, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_steps)

    # test dataset loss
    total_test_step = 0
    total_test_loss = 0
    test_acc = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            output = model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            acc = (output.argmax(1) == targets).sum()
            test_acc += acc

    print('Total loss on test data {}'.format(total_test_loss))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', test_acc / test_data_size, total_test_step)
    total_test_step += 1

writer.close()

# Save model
# torch.save(model.state_dict(), 'model.pth')
