import tensorflow as tf
import numpy as np
import cifar_data_loader

(train_images, train_labels, test_images, test_labels, mean_image) = cifar_data_loader.load_data()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
print(mean_image.shape)

graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    labels = tf.placeholder(tf.int32, shape=(None), name="labels")

    layer1_weights = tf.get_variable("layer1_weights", [3, 3, 3, 64], initializer=tf.contrib.layers.variance_scaling_initializer())
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

    layer5_weights = tf.get_variable("layer5_weights", [3, 3, 128, 256], initializer=tf.contrib.layers.variance_scaling_initializer())
    layer5_bias = tf.Variable(tf.zeros([256]))
    layer5_conv = tf.nn.conv2d(pool2, filter=layer5_weights, strides=[1,1,1,1], padding='SAME')
    layer5_out = tf.nn.relu(layer5_conv + layer5_bias)
    
    layer6_weights = tf.get_variable("layer6_weights", [3, 3, 256, 256], initializer=tf.contrib.layers.variance_scaling_initializer())
    layer6_bias = tf.Variable(tf.zeros([256]))
    layer6_conv = tf.nn.conv2d(layer5_out, filter=layer6_weights, strides=[1,1,1,1], padding='SAME')
    layer6_out = tf.nn.relu(layer6_conv + layer6_bias)

    layer7_weights = tf.get_variable("layer7_weights", [3, 3, 256, 256], initializer=tf.contrib.layers.variance_scaling_initializer())
    layer7_bias = tf.Variable(tf.zeros([256]))
    layer7_conv = tf.nn.conv2d(layer6_out, filter=layer7_weights, strides=[1,1,1,1], padding='SAME')
    layer7_out = tf.nn.relu(layer7_conv + layer7_bias)

    pool3 = tf.nn.max_pool(layer7_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    layer8_weights = tf.get_variable("layer8_weights", [3, 3, 256, 512], initializer=tf.contrib.layers.variance_scaling_initializer())
    layer8_bias = tf.Variable(tf.zeros([512]))
    layer8_conv = tf.nn.conv2d(pool3, filter=layer8_weights, strides=[1,1,1,1], padding='SAME')
    layer8_out = tf.nn.relu(layer8_conv + layer8_bias)
    
    layer9_weights = tf.get_variable("layer9_weights", [3, 3, 512, 512], initializer=tf.contrib.layers.variance_scaling_initializer())
    layer9_bias = tf.Variable(tf.zeros([512]))
    layer9_conv = tf.nn.conv2d(layer8_out, filter=layer9_weights, strides=[1,1,1,1], padding='SAME')
    layer9_out = tf.nn.relu(layer9_conv + layer9_bias)

    layer10_weights = tf.get_variable("layer10_weights", [3, 3, 512, 512], initializer=tf.contrib.layers.variance_scaling_initializer())
    layer10_bias = tf.Variable(tf.zeros([512]))
    layer10_conv = tf.nn.conv2d(layer9_out, filter=layer10_weights, strides=[1,1,1,1], padding='SAME')
    layer10_out = tf.nn.relu(layer10_conv + layer10_bias)
    
    pool4 = tf.nn.max_pool(layer10_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    shape = pool4.shape.as_list()
    newShape = shape[1] * shape[2] * shape[3]
    reshaped_pool4 = tf.reshape(pool4, [-1, newShape])

    fc1_weights = tf.get_variable("layer11_weights", [newShape, 4096], initializer=tf.contrib.layers.variance_scaling_initializer())
    fc1_bias =  tf.Variable(tf.zeros([4096]))
    fc1_out = tf.nn.relu(tf.matmul(reshaped_pool4, fc1_weights) + fc1_bias)

    fc2_weights = tf.get_variable("layer12_weights", [4096, 10], initializer=tf.contrib.layers.xavier_initializer())
    fc2_bias =  tf.Variable(tf.zeros([10]))
    logits = tf.matmul(fc1_out, fc2_weights) + fc2_bias

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    #Add a few nodes to calculate accuracy and optionally retrieve predictions
    predictions = tf.nn.softmax(logits)
    correct_prediction = tf.equal(labels, tf.argmax(predictions, 1, output_type=tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        num_steps = 10000
        batch_size = 100
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_images = train_images[offset:(offset + batch_size)]
            batch_labels = train_labels[offset:(offset + batch_size)]
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







