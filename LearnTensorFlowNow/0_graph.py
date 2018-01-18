import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_images = mnist.train.images;
train_labels = mnist.train.labels

graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, shape=(100, 784),name="input")
    labels = tf.placeholder(tf.float32, shape=(100, 10),name="labels")

    with tf.name_scope("layer1"):
        layer1_weights = tf.Variable(tf.random_normal([784, 10]), name="layer1_weights")
        layer1_bias = tf.Variable(tf.zeros([10]), name="layer1_bias")

        with tf.name_scope("logits"):
            logits = tf.add(tf.matmul(input, layer1_weights), layer1_bias, name="logits")

        with tf.name_scope("cross_entropy"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels), name="cross_entropy")

    with tf.name_scope("optimizer"):
        learning_rate = 0.01
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name="optimizer")

    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter("/test/tensorboard")
        writer.add_graph(session.graph)

        tf.global_variables_initializer().run()

        num_steps = 1000
        batch_size = 100
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_images = train_images[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {input: batch_images, labels: batch_labels}

            o, c, = session.run([optimizer, cost], feed_dict=feed_dict)
            print("Cost: ", c)


