import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_images = mnist.train.images;
train_labels = mnist.train.labels
test_images = mnist.test.images;
test_labels = mnist.test.labels

graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, shape=(None, 784), name="input")
    labels = tf.placeholder(tf.float32, shape=(None, 10), name="labels")

    with tf.name_scope("layer1"):
        layer1_weights = tf.Variable(tf.random_normal([784, 500]), name="layer1_weights")
        layer1_bias = tf.Variable(tf.zeros([500]), name="layer1_bias")
        layer1_output = tf.nn.relu(tf.matmul(input, layer1_weights) + layer1_bias, name="layer1_output")

    with tf.name_scope("layer2"):
        layer2_weights = tf.Variable(tf.random_normal([500, 500]), name="layer2_weights")
        layer2_bias = tf.Variable(tf.zeros([500]), name="layer2_bias")
        layer2_output = tf.nn.relu(tf.matmul(layer1_output, layer2_weights) + layer2_bias, name="layer2_output")

    with tf.name_scope("layer3"):
        layer3_weights = tf.Variable(tf.random_normal([500, 10]), name="layer3_weights")
        layer3_bias = tf.Variable(tf.zeros([10]), name="layer3_weights")
        logits = tf.add(tf.matmul(layer2_output, layer3_weights), layer3_bias, name="logits")

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels), name="cost")

    with tf.name_scope("optimizer"):
        learning_rate = 0.0001
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    #Add a few nodes to calculate accuracy and optionally retrieve predictions
    predictions = tf.nn.softmax(logits, name="predictions")

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter("/test/tensorboard")
        writer.add_graph(session.graph)
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
            batch_images = test_images[offset:(offset + batch_size), :]
            batch_labels = test_labels[offset:(offset + batch_size), :]
            feed_dict = {input: batch_images, labels: batch_labels}

            _, c, acc = session.run([optimizer, cost, accuracy], feed_dict=feed_dict)
            total_cost = total_cost + c
            total_accuracy = total_accuracy + acc

        print("Test Cost: ", total_cost / num_test_batches)
        print("Test accuracy: ", total_accuracy * 100.0 / num_test_batches, "%")

