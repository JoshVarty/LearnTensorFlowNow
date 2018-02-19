import tensorflow as tf
import numpy as np
import shutil
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_images = np.reshape(mnist.train.images, (-1, 28, 28, 1))
train_labels = mnist.train.labels
test_images = np.reshape(mnist.test.images, (-1, 28, 28, 1))
test_labels = mnist.test.labels

graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    labels = tf.placeholder(tf.float32, shape=(None, 10))

    layer1_weights = tf.get_variable("layer1_weights", [3, 3, 1, 64], initializer=tf.contrib.layers.variance_scaling_initializer())
    layer1_bias = tf.Variable(tf.zeros([64]))
    layer1_conv = tf.nn.conv2d(input, filter=layer1_weights, strides=[1,1,1,1], padding='SAME')
    layer1_out = tf.nn.relu(layer1_conv + layer1_bias)
    
    layer2_weights = tf.get_variable("layer2_weights", [3, 3, 64, 64], initializer=tf.contrib.layers.variance_scaling_initializer())
    layer2_bias = tf.Variable(tf.zeros([64]))
    layer2_conv = tf.nn.conv2d(layer1_out, filter=layer2_weights, strides=[1,1,1,1], padding='SAME')
    layer2_out = tf.nn.relu(layer2_conv + layer2_bias)

    pool1 = tf.nn.max_pool(layer2_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    layer3_weights = tf.get_variable("layer3_weights", [3, 3, 64, 128], initializer=tf.contrib.layers.variance_scaling_initializer())
    layer3_bias = tf.Variable(tf.zeros([128]))
    layer3_conv = tf.nn.conv2d(pool1, filter=layer3_weights, strides=[1,1,1,1], padding='SAME')
    layer3_out = tf.nn.relu(layer3_conv + layer3_bias)

    layer4_weights = tf.get_variable("layer4_weights", [3, 3, 128, 128], initializer=tf.contrib.layers.variance_scaling_initializer())
    layer4_bias = tf.Variable(tf.zeros([128]))
    layer4_conv = tf.nn.conv2d(layer3_out, filter=layer4_weights, strides=[1,1,1,1], padding='SAME')
    layer4_out = tf.nn.relu(layer4_conv + layer4_bias)

    pool2 = tf.nn.max_pool(layer4_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    shape = pool2.shape.as_list()
    fc = shape[1] * shape[2] * shape[3]
    reshape = tf.reshape(pool2, [-1, fc])
    fully_connected_weights = tf.get_variable("fully_connected_weights", [fc, 10], initializer=tf.contrib.layers.xavier_initializer())
    fully_connected_bias = tf.Variable(tf.zeros([10]))
    logits = tf.matmul(reshape, fully_connected_weights) + fully_connected_bias

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    learning_rate = 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    #Add a few nodes to calculate accuracy and optionally retrieve predictions
    predictions = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        num_steps = 5000
        batch_size = 100
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_images = train_images[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {input: batch_images, labels: batch_labels}

            _, c, acc = session.run([optimizer, cost, accuracy], feed_dict=feed_dict)

            if step % 100 == 0: 
                print("Cost: ", c)
                print("Accuracy: ", acc * 100.0, "%")

        #Test 
        num_test_batches = int(len(test_images) / 100)
        total_accuracy = 0
        total_cost = 0
        for step in range(num_test_batches):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_images = test_images[offset:(offset + batch_size)]
            batch_labels = test_labels[offset:(offset + batch_size)]
            feed_dict = {input: batch_images, labels: batch_labels}

            c, acc = session.run([cost, accuracy], feed_dict=feed_dict)
            total_cost = total_cost + c
            total_accuracy = total_accuracy + acc

        print("Test Cost: ", total_cost / num_test_batches)
        print("Test accuracy: ", total_accuracy * 100.0 / num_test_batches, "%")

