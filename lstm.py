import datetime

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score

tf.reset_default_graph()

X = np.load('data/recruiting/idsMatrix.npy')
Y = np.load('data/recruiting/idsLabels.npy')

RANDOM_SEED = 42
max_seq_length = 250
batch_size = 300
lstm_units = 64
num_classes = 2
iterations = 100000
num_dimensions = 300

def get_train_test_data():
    """ Read the iris data set and split them into training and test sets """
    return train_test_split(X, Y, train_size=batch_size, test_size=batch_size, random_state=RANDOM_SEED)

word_vectors = np.load('data/recruiting/wordVectors.npy')
words_list = np.load('data/recruiting/wordsList.npy')

labels = tf.placeholder(tf.float32, [batch_size, num_classes])
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_length])

data = tf.Variable(tf.zeros([batch_size, max_seq_length, num_dimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(word_vectors, input_data)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(logdir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
   #Next Batch of reviews
   train_X, test_X, train_y, test_y = get_train_test_data()
   sess.run(optimizer, {input_data: train_X, labels: train_y})

   #Write summary to Tensorboard
   if (i % 50 == 0):
       train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                sess.run(prediction, feed_dict={input_data: train_X, labels: train_y}))
       test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                               sess.run(prediction, feed_dict={input_data: test_X, labels: test_y}))
       # roc = roc_auc_score(test_y, axis=1), sess.run(prediction, feed_dict={input_data: test_X, labels: test_y}))

       summary = sess.run(merged, {input_data: test_X, labels: test_y})
       writer.add_summary(summary, i)
       # print(classification_report(test_y, sess.run(prediction, feed_dict={input_data: test_X, labels: test_y})))
       # print(roc)
       # print('')

   #Save the network every 10,000 training iterations
   if (i % 500 == 0 and i != 0):
       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)

writer.close()