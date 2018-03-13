import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

test_images = np.reshape(mnist.test.images, (-1, 28, 28, 1))
test_labels = mnist.test.labels

with tf.Session() as session:
    saver = tf.train.import_meta_graph('/tmp/vggnet/vgg_net.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    #Now we test our restored model just as before 
    batch_size = 100
    num_test_batches = int(len(test_images) / 100)
    total_accuracy = 0
    total_cost = 0
    for step in range(num_test_batches):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_images = test_images[offset:(offset + batch_size)]
        batch_labels = test_labels[offset:(offset + batch_size)]
        feed_dict = {input: batch_images, labels: batch_labels}

        c, acc = session.run(['cost:0', 'accuracy:0'], feed_dict=feed_dict)
        total_cost = total_cost + c
        total_accuracy = total_accuracy + acc

    print("Test Cost: ", total_cost / num_test_batches)
    print("Test accuracy: ", total_accuracy * 100.0 / num_test_batches, "%")