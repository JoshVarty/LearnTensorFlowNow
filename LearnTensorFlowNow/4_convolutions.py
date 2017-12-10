import tensorflow as tf
import numpy as np
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

    layer1_weights = tf.Variable(tf.random_normal([3, 3, 1, 64]))
    layer1_bias = tf.Variable(tf.zeros([64]))
    net = tf.nn.conv2d(input, filter=layer1_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer1_bias
    
    layer2_weights = tf.Variable(tf.random_normal([3, 3, 64, 64]))
    layer2_bias = tf.Variable(tf.zeros([64]))
    net = tf.nn.conv2d(net, filter=layer2_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer2_bias

    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    layer3_weights = tf.Variable(tf.random_normal([3, 3, 64, 128]))
    layer3_bias = tf.Variable(tf.zeros([128]))
    net = tf.nn.conv2d(net, filter=layer3_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer3_bias
    
    layer4_weights = tf.Variable(tf.random_normal([3, 3, 128, 128]))
    layer4_bias = tf.Variable(tf.zeros([128]))
    net = tf.nn.conv2d(net, filter=layer4_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer4_bias

    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    shape = net.shape.as_list()
    fc = shape[1] * shape[2] * shape[3]
    reshape = tf.reshape(net, [-1, fc])
    fc_weights = tf.Variable(tf.random_normal([fc, 10]))
    fc_bias = tf.Variable(tf.zeros([10]))
    logits = tf.matmul(reshape, fc_weights) + fc_bias

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    learning_rate = 0.0000001
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
            x = batch_images.shape
            feed_dict = {input: batch_images, labels: batch_labels}

            _, c, acc = session.run([optimizer, cost, accuracy], feed_dict=feed_dict)
            total_cost = total_cost + c
            total_accuracy = total_accuracy + acc

        print("Test Cost: ", total_cost / num_test_batches)
        print("Test accuracy: ", total_accuracy * 100.0 / num_test_batches, "%")

