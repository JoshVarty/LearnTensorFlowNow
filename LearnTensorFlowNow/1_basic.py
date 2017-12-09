import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_images = mnist.train.images;
train_labels = mnist.train.labels
test_images = mnist.test.images;
test_labels = mnist.test.labels

graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, shape=(None, 784))
    labels = tf.placeholder(tf.float32, shape=(None, 10))

    layer1_weights = tf.Variable(tf.random_normal([784, 10]))
    layer1_bias = tf.Variable(tf.zeros([10]))

    logits = tf.matmul(input, layer1_weights) + layer1_bias
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        num_steps = 1000
        batch_size = 100
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_images = train_images[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {input: batch_images, labels: batch_labels}

            _, c, = session.run([optimizer, cost], feed_dict=feed_dict)
            print("Cost: ", c)

