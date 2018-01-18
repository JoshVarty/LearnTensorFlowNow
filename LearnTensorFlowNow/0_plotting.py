import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import operator

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_images = mnist.train.images;
train_labels = mnist.train.labels

lookup = {}

for idx in range(len(train_images)):
    if idx % 1000 == 0:
        print(idx)
    image = train_images[idx]
    currentCount = 0 
    for pixel in image:
        if pixel != 0:
            currentCount = currentCount + 1

    lookup[idx] = currentCount


sorted_x = sorted(lookup.items(), key=operator.itemgetter(1))

for key, value in sorted_x[:10]:
    plt.imshow(train_images[value].reshape(28,28), cmap="Greys")
    plt.show()





x = input("done")


