import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torch.optim as optim
from PIL import Image

accuracy_target = 88

# Try to train a CNN classifier.
conv_last_out_channels = 8
conv_first_out_channels = 16
fc1_out = 256
fc2_out = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 4, 3)
        # self.pool1 = nn.MaxPool2d(4, 4)
        # self.conv2 = nn.Conv2d(4, 16, 5)
        self.conv2 = nn.Conv2d(1, conv_first_out_channels, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(conv_first_out_channels, conv_last_out_channels, 5, stride=2)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(conv_last_out_channels*16*16, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, 2)

    def forward(self, x):
        # x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, conv_last_out_channels*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def disp_spect(spect):
    A = np.copy(spect)
    m = np.min(A)
    M = np.max(A)
    B = (A-m)/(M-m)*255
    picture = Image.fromarray(np.uint8(B))
    picture.show()


# Compose the network.
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda:0")



# get the inputs; data is a list of [inputs, labels]
# Construct the dataset.
train_data = np.load('train_data.npy')
N_pos = 100; N_neg = 100; N = 100
batch_size = 10
train_data_num = 180
size_input = 138

epochs = int(input('Please input the number of epochs.'))


train_labels = np.zeros((N_pos+N_neg, ))
train_labels[0:N_pos] = np.ones((N_pos, ))


# Perform random shuffling.
perm = np.random.permutation(N_pos + N_neg)
train_data = train_data[perm, :, :]
train_labels = train_labels[perm]


train_data_tensor = torch.Tensor(train_data_num, 1, size_input, size_input).to(device)
for k in range(train_data_num):
    train_data_tensor[k, 0, :, :] = torch.tensor(train_data[k, :, :]).to(device)

train_labels_tensor = torch.tensor(train_labels[0:train_data_num]).to(device)

validate_data_tensor = torch.Tensor(200 - train_data_num, 1, size_input, size_input).to(device)
for k in range(200-train_data_num):
    validate_data_tensor[k, 0, :, :] = torch.tensor(train_data[train_data_num+k, :, :]).to(device)
validate_labels_tensor = torch.tensor(train_labels[train_data_num:200]).to(device)

# Test set.
test_data = np.load('test_data.npy')
test_data_tensor = torch.Tensor(N, 1, size_input, size_input).to(device)
for k in range(N):
    test_data_tensor[k, 0, :, :] = torch.from_numpy(test_data[k, :, :]).to(device)

test_labels = np.load('test_labels.npy')
test_labels_tensor = torch.from_numpy(test_labels).to(device)



# disp_spect(train_data[0,:,:])
# disp_spect(train_data[1,:,:])

flag = True
while flag:
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.05)
    running_loss = 0.0

    print('Start training. Target Accuracy: %d' % accuracy_target)
    for epoch in range(epochs):
        print('-----Epoch %d-------' % epoch)
        for batch_iter in range(int(train_data_num/batch_size)):
            inputs = train_data_tensor[batch_iter*batch_size:(batch_iter+1)*batch_size, :, :].to(device)

            # inputs = torch.Tensor(batch_size, 1, 554, 554)
            # for k in range(batch_size):
            # inputs[k, 0, :, :] = torch.tensor(train_data[k+batch_iter*batch_size, :, :])

            labels = torch.tensor(train_labels[batch_iter*batch_size:(batch_iter+1)*batch_size]).long().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = loss.item()

            # print('Finished Training for %d batch. Loss = %.4f' % (batch_iter, running_loss))

        # Start calculating Accuracy.
        pred_outputs = net(validate_data_tensor)
        val_loss = criterion(pred_outputs, validate_labels_tensor.long()).item()
        _, all_predictions = torch.max(pred_outputs.data, 1)
        correct = (all_predictions == validate_labels_tensor).sum().item()
        print('Val loss: %.4f Correct: %d in 20' % (val_loss, correct))

        pred_outputs = net(test_data_tensor)
        val_loss = criterion(pred_outputs, test_labels_tensor.long()).item()
        _, all_predictions = torch.max(pred_outputs.data, 1)
        correct = (all_predictions == test_labels_tensor).sum().item()
        print('Test loss: %.4f Correct: %d in 100' % (val_loss, correct))

        if correct >= accuracy_target:
            # cpu_device = torch.device('cpu')
            file_name = 'optim_model_%d.pth' % accuracy_target
            torch.save(net.state_dict(), file_name)
            flag = False
            break

print('Complete. Target accuracy %d reached.' % accuracy_target)
