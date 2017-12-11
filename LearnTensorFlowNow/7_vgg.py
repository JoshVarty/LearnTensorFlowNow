import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_images = np.reshape(mnist.train.images, (-1, 28, 28, 1))
train_labels = mnist.train.labels
test_images = np.reshape(mnist.test.images, (-1, 28, 28, 1))
test_labels = mnist.test.labels

def bias_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))

def relu_weight_layer(name, shape):
    return tf.get_variable(name, shape, initializer = tf.contrib.layers.variance_scaling_initializer())

def softmax_weight_layer(name, shape):
    return tf.get_variable(name, shape, initializer = tf.contrib.layers.xavier_initializer())

graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    labels = tf.placeholder(tf.float32, shape=(None, 10))

    net = tf.image.resize_image_with_crop_or_pad(input, target_height=32, target_width=32)

    layer1_weights = relu_weight_layer("w_layer1", [3, 3, 1, 64])
    layer1_bias = bias_variable("b_layer1", [64])
    net = tf.nn.conv2d(net, filter=layer1_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer1_bias
    
    layer2_weights = relu_weight_layer("w_layer2", [3, 3, 64, 64])
    layer2_bias = bias_variable("b_layer2", [64])
    net = tf.nn.conv2d(net, filter=layer2_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer2_bias

    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    layer3_weights = relu_weight_layer("w_layer3", [3, 3, 64, 128])
    layer3_bias = bias_variable("b_layer3", [128])
    net = tf.nn.conv2d(net, filter=layer3_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer3_bias
    
    layer4_weights = relu_weight_layer("w_layer4", [3, 3, 128, 128])
    layer4_bias = bias_variable("b_layer4", [128])
    net = tf.nn.conv2d(net, filter=layer4_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer4_bias

    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    layer5_weights = relu_weight_layer("w_layer5", [3, 3, 128, 256])
    layer5_bias = bias_variable("b_layer5", [256])
    net = tf.nn.conv2d(net, filter=layer5_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer5_bias
    
    layer6_weights = relu_weight_layer("w_layer6", [3, 3, 256, 256])
    layer6_bias = bias_variable("b_layer6", [256])
    net = tf.nn.conv2d(net, filter=layer6_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer6_bias

    layer7_weights = relu_weight_layer("w_layer7", [3, 3, 256, 256])
    layer7_bias = bias_variable("b_layer7", [256])
    net = tf.nn.conv2d(net, filter=layer7_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer7_bias

    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    layer8_weights = relu_weight_layer("w_layer8", [3, 3, 256, 512])
    layer8_bias = bias_variable("b_layer8", [512])
    net = tf.nn.conv2d(net, filter=layer8_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer8_bias
    
    layer9_weights = relu_weight_layer("w_layer9", [3, 3, 512, 512])
    layer9_bias = bias_variable("b_layer9", [512])
    net = tf.nn.conv2d(net, filter=layer9_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer9_bias

    layer10_weights = relu_weight_layer("w_layer10", [3, 3, 512, 512])
    layer10_bias = bias_variable("b_layer10", [512])
    net = tf.nn.conv2d(net, filter=layer10_weights, strides=[1,1,1,1], padding='SAME')
    net = tf.nn.relu(net) + layer10_bias
    
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    shape = net.shape.as_list()
    fc = shape[1] * shape[2] * shape[3]
    net = tf.reshape(net, [-1, fc])
    fc1_weights = softmax_weight_layer("w_fc1", [fc, 4096])
    fc1_bias =  bias_variable("b_fc1", [4096])
    net = tf.matmul(net, fc1_weights) + fc1_bias

    fc2_weights = softmax_weight_layer("w_fc2", [4096, 10])
    fc2_bias =  bias_variable("b_fc2", [10])
    logits = tf.matmul(net, fc2_weights) + fc2_bias

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    #Add a few nodes to calculate accuracy and optionally retrieve predictions
    predictions = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        num_steps = 1000
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



