# Lab 9 XOR
import tensorflow as tf
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
plt.ion()

def DisplayErrorPlot(train_error):
  plt.figure(1)
  plt.clf()
  plt.plot(range(len(train_error)), train_error, 'b', label='Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss function')
  plt.legend()
  plt.draw()
  input('Press Enter to exit.')

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.01

x_data = []
y_data = []
losses = []

dataFile = open("./L10.csv","r")
csvReader = csv.reader(dataFile)

for row in csvReader:
	aa = row
	tx = []
	ty = []
	for i in range(5):
		tx.append(aa[i])
	ty.append(aa[5])
	x_data.append(tx)
	y_data.append(ty)

dataFile.close()

inputN = 5

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 5])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([5, 1]), name='weight1')
b1 = tf.Variable(tf.random_normal([1]), name='bias1')
hypothesis = tf.add(tf.matmul(X, W1), b1)

# cost/loss function
cost = tf.reduce_mean(tf.pow(hypothesis - Y, 2))
#cost = -tf.reduce_mean(1 * Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.initialize_all_variables())

    for step in range(50001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        losses.append(sess.run(cost, feed_dict={ X: x_data, Y: y_data}))
        if step % 1000 == 0:
            print("Step:",step, ", Loss:",sess.run(cost, feed_dict={ X: x_data, Y: y_data}), ", Weight:", sess.run([W1]))

    # Accuracy report
    h, c = sess.run([hypothesis, cost],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ",    c)

    outputfile = open("./output.csv","w")
    wr = csv.writer(outputfile)
    for i in range(len(losses)) :
        wr.writerow(str(losses[i]))

    outputfile.close()
    print(W1.eval())
    print(b1.eval())