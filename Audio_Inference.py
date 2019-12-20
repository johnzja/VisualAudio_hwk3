import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

# Use trained neural network.
conv_last_out_channels = 8
conv_first_out_channels = 16
fc1_out = 256
fc2_out = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2 = nn.Conv2d(1, conv_first_out_channels, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(conv_first_out_channels, conv_last_out_channels, 5, stride=2)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(conv_last_out_channels*16*16, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, 2)

    def forward(self, x):
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, conv_last_out_channels*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load state dicts from model pth file.
model = torch.load('optim_model_88.pth', map_location='cpu')
net = Net()
criterion = nn.CrossEntropyLoss()
net.load_state_dict(model)

# Perform Inference using test_data.npy and test_labels.npy. Print all the results.
# Test set.
N_pos = 100; N_neg = 100; N = 100
batch_size = 5
size_input = 138

test_data = np.load('test_data.npy')
test_data_tensor = torch.Tensor(N, 1, size_input, size_input)
for k in range(N):
    test_data_tensor[k, 0, :, :] = torch.from_numpy(test_data[k, :, :])

test_labels = np.load('test_labels.npy')
test_labels_tensor = torch.from_numpy(test_labels)


# Perform prediction
pred_outputs = net(test_data_tensor)
val_loss = criterion(pred_outputs, test_labels_tensor.long()).item()
_, all_predictions = torch.max(pred_outputs.data, 1)
correct = (all_predictions == test_labels_tensor).sum().item()
print('Test loss: %.4f Correct: %d in 100' % (val_loss, correct))

