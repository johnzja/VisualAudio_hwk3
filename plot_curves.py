import numpy as np
import matplotlib.pyplot as plt

train_loss_curve, val_loss_curve, test_loss_curve = np.load('loss_curves.npy', allow_pickle=True)
train_acc_curve, val_acc_curve, test_acc_curve = np.load('acc_curves.npy', allow_pickle=True)

epochs = train_loss_curve.size
xs = np.array([x+1 for x in range(epochs)])

plt.figure()
plt.plot(xs, train_loss_curve, label='Train loss', color='green')
plt.plot(xs, val_loss_curve, label='Validate loss', color='red')
plt.plot(xs, test_loss_curve, label='Test loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(xs, train_acc_curve, label='Train acc', color='green')
plt.plot(xs, val_acc_curve, label='Validate acc', color='red')
plt.plot(xs, test_acc_curve, label='Test acc', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

