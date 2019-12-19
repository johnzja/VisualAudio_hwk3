import wave
import numpy as np
import pylab as plt

from pylab import *
import torch
import torchvision
import torch.optim as optim
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import cv2

# Open the wave file.
wav_file = wave.open(r'audio.wav', 'rb')
params = wav_file.getparams()
nchannels, samplewidth, framerate, nframes = params[:4]
str_data = wav_file.readframes(nframes)
wav_file.close()

wave_data = np.fromstring(str_data, dtype=np.short)
wave_data = wave_data * 1.0/max(abs(wave_data))
wave_data.shape = (nframes,)
# wave_data = wave_data.T
time = np.arange(0, nframes)/framerate

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(time, wave_data)
plt.xlabel('time')

# Perform spectrogram analysis.
Fs = framerate
plt.subplot(2, 1, 2)
spectrum, freqs, ts, fig = plt.specgram(wave_data, NFFT=4096, Fs=Fs, noverlap=2048)
spect_data = spectrum

# HERE: spectrum data get.
spect_data = spect_data[0:1000, :]
maxim = np.max(np.max(np.abs(spect_data), axis=0), axis=0)
spect_data = spect_data * 1.0/maxim

plt.show()

# Plot the spect_data.
def disp_spect(spect):
    A = np.copy(np.log(spect))
    m = np.min(A)
    M = np.max(A)
    B = (A-m)/(M-m)*255
    picture = Image.fromarray(np.uint8(B))
    # picture.rotate(90)
    picture.show()

def spect_resize(spect):
    A = np.copy(spect)
    B = cv2.resize(A.astype('float'), (554,554), interpolation=cv2.INTER_CUBIC)
    return B

disp_spect(spect_data)



# Try to train a CNN classifier.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(4, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5, stride=2)
        self.pool3 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(32*16*16, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# Compose the network.
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
running_loss = 0.0

# get the inputs; data is a list of [inputs, labels]
inputs = torch.Tensor(1, 1, 554, 554)
inputs[0, 0, :, :] = torch.tensor(spect_resize(spect_data))
labels = torch.Tensor(1).long()
labels[0] = torch.tensor(1)

# zero the parameter gradients
optimizer.zero_grad()

# forward + backward + optimize
outputs = net(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

# print statistics
running_loss += loss.item()

print('Finished Training for 1 picture.')









