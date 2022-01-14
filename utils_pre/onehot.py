import numpy as np

num_classes = 10
labels = np.load('./cifar_color_label.npy')

labels_onehot = np.zeros((labels.shape[0], num_classes))
for i in range(labels.shape[0]):
    labels_onehot[i,labels[i][0]] = 1.

np.save('cifar_color_label_onehot.npy', labels_onehot)
