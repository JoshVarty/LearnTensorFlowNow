import numpy as np
import matplotlib.pyplot as plt
import cifar_data_loader
import random

(train_images, train_labels, test_images, test_labels, mean_image) = cifar_data_loader.load_data()


fig = plt.figure(figsize=(32,32))
columns = 4
rows = 1

for i in range(1, columns*rows +1):
    idx = random.randint(0, 50000-1)
    img = train_images[idx]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)

plt.show()



