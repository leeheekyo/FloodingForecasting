# Lab 9 XOR
import tensorflow as tf
import numpy as np
import random
from datetime import datetime

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

#choose filename
filename="data_final"

N = 6

times = 0

xlist = []
ylist = []
x_test = []
y_test = []

def calc1(mat1, sub1):
  sumval1 = 0
  total1 = 0
  sumval0 =0
  total0 = 0
  ilist = []
  x_test_len = len(x_test)
  for i in range(x_test_len):
    z = np.dot(x_test[i], mat1)
    z = z+sub1
    if(y_test[i][0]==1 ):
      if(z>0):
        sumval1+=1
      total1+=1
    elif(y_test[i][0]==0):
      if(z<0):
        sumval0+=1
      total0+=1
  #print(sumval, x_test_len)
  print("sumval1 : " + str(sumval1) +", total1 : " + str(total1))
  print("sumval0 : " + str(sumval0) +", total0 : " + str(total0))
  print("sumval  : " + str(sumval1+sumval0) +", total   : " + str(x_test_len))

def calc2(mat1, sub1, mat2, sub2):
  sumval1 = 0
  total1 = 0
  sumval0 =0
  total0 = 0
  ilist = []
  x_test_len = len(x_test)
  for i in range(x_test_len):
    z = np.dot(x_test[i], mat1)
    z = z+sub1
    for j in range(N): 
      if z[j]>0 :
        z[j]=1
      else :
        z[j]=0
    z = np.dot(z, mat2)
    z = z+sub2   

    if(y_test[i][0]==1 ):
      if(z>0):
        sumval1+=1
      total1+=1
    elif(y_test[i][0]==0):
      if(z<0):
        sumval0+=1
      total0+=1
  #print(sumval, x_test_len)
  print("sumval1 : " + str(sumval1) +", total1 : " + str(total1))
  print("sumval0 : " + str(sumval0) +", total0 : " + str(total0))
  print("sumval  : " + str(sumval1+sumval0) +", total   : " + str(x_test_len))

def calc3(mat1, sub1, mat2, sub2, mat3, sub3):
  sumval1 = 0
  total1 = 0
  sumval0 =0
  total0 = 0
  ilist = []
  x_test_len = len(x_test)
  for i in range(x_test_len):
    z = np.dot(x_test[i], mat1)
    z = z+sub1
    for j in range(N): 
      if z[j]>0 :
        z[j]=1
      else :
        z[j]=0
    z = np.dot(z, mat2)
    z = z+sub2
    for j in range(N): 
      if z[j]>0 :
        z[j]=1
      else :
        z[j]=0
    z = np.dot(z, mat3)
    z = z+sub3  

    if(y_test[i][0]==1 ):
      if(z>0):
        sumval1+=1
      total1+=1
    elif(y_test[i][0]==0):
      if(z<0):
        sumval0+=1
      total0+=1
  #print(sumval, x_test_len)
  print("sumval1 : " + str(sumval1) +", total1 : " + str(total1))
  print("sumval0 : " + str(sumval0) +", total0 : " + str(total0))
  print("sumval  : " + str(sumval1+sumval0) +", total   : " + str(x_test_len))

###############################################
################# print array #################
###############################################
def printArr(arr):
  pval=""
  if(str(type(arr[0]))=="<class 'numpy.ndarray'>"):
    i_length = len(arr)
    j_length = len(arr[0])
    pval+="["
    for i in range(i_length):
      pval+="["
      for j in range(j_length-1):
        pval+=str(arr[i][j])
        pval+=","
      pval+=str(arr[i][j_length-1])
      pval+="]"
      if(i!=i_length-1):
        pval+=","
    pval+="]"
  else :
    i_length = len(arr)
    pval+="["
    for i in range(i_length):
      pval+=str(arr[i])
      if(i!=i_length-1):
        pval+=","
    pval+="]"
  print(pval)


###############################################
################ data learning ################
###############################################
def learning3(x_data, y_data):
  x_data = np.array(x_data, dtype=np.float32)
  y_data = np.array(y_data, dtype=np.float32)
  xl_data = np.array(x_real, dtype=np.float32)
  yl_data = np.array(y_real, dtype=np.float32)
  
  X = tf.placeholder(tf.float32, [None, N])
  Y = tf.placeholder(tf.float32, [None, 1])
  
  W1 = tf.Variable(tf.random_normal([N, N]), name='weight1')
  b1_tmp = []
  for ii in range(N):
    b1_tmp.append(c)
  b1 = tf.Variable(b1_tmp, name='bias1')
  layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
  
  W2 = tf.Variable(tf.random_normal([N, N]), name='weight2')
  b2 = tf.Variable(tf.random_normal([N]), name='bias2')
  layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
  
  W3 = tf.Variable(tf.random_normal([N, 1]), name='weight3')
  b3 = tf.Variable(tf.random_normal([1]), name='bias3')
  hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3)
  
  # cost/loss function
  cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
  #cost = tf.reduce_mean(tf.square(hypothesis - Y))
  
  train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
  
  # Accuracy computation
  # True if hypothesis>0.5 else False
  predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
  
  # Launch graph
  with tf.Session() as sess:
      # Initialize TensorFlow variables
      sess.run(tf.global_variables_initializer())
  
      #maxA = 0
      time = 0
      #count = 0
      for step in range(times):
          time+=1
          sess.run(train, feed_dict={X: x_data, Y: y_data})
      h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: xl_data, Y: yl_data})
      print("\nAccuracy: ", a, " time : ", time)
      calc3(W1.eval(),b1.eval(),W2.eval(),b2.eval(),W3.eval(),b3.eval())

def learning2(x_data, y_data):
  x_data = np.array(x_data, dtype=np.float32)
  y_data = np.array(y_data, dtype=np.float32)
  xl_data = np.array(x_real, dtype=np.float32)
  yl_data = np.array(y_real, dtype=np.float32)
  
  X = tf.placeholder(tf.float32, [None, N])
  Y = tf.placeholder(tf.float32, [None, 1])
  
  W1 = tf.Variable(tf.random_normal([N, N]), name='weight1')
  b1 = tf.Variable(tf.random_normal([N]), name='bias1')
  layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
  
  W2 = tf.Variable(tf.random_normal([N, 1]), name='weight2')
  b2 = tf.Variable(tf.random_normal([1]), name='bias2')
  hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)
  
  # cost/loss function
  cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
  #cost = tf.reduce_mean(tf.square(hypothesis - Y))
  
  train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
  
  # Accuracy computation
  # True if hypothesis>0.5 else False
  predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
  
  # Launch graph
  with tf.Session() as sess:
      # Initialize TensorFlow variables
      sess.run(tf.global_variables_initializer())
      #maxA = 0
      time = 0
      #count = 0
      for step in range(times):
          time+=1
          sess.run(train, feed_dict={X: x_data, Y: y_data})
          if step%100 == 0 :
            h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
            print("\nAccuracy: ", a, " time : ", time)
      h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: xl_data, Y: yl_data})
      print("\nAccuracy: ", a, " time : ", time)
      calc2(W1.eval(),b1.eval(),W2.eval(),b2.eval())

def learning1(x_data, y_data):
  x_data = np.array(x_data, dtype=np.float32)
  y_data = np.array(y_data, dtype=np.float32)
  xl_data = np.array(x_real, dtype=np.float32)
  yl_data = np.array(y_real, dtype=np.float32)
  
  X = tf.placeholder(tf.float32, [None, N])
  Y = tf.placeholder(tf.float32, [None, 1])
  
  W1 = tf.Variable(tf.random_normal([N, 1]), name='weight1')
  b1 = tf.Variable(tf.random_normal([1]), name='bias1')
  hypothesis = tf.sigmoid(tf.matmul(X, W1) + b1)
  
  # cost/loss function
  cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y+1e-10) * tf.log(1 - hypothesis+1e-10))
  #cost = tf.reduce_mean(tf.square(hypothesis - Y))
  
  train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
  
  # Accuracy computation
  # True if hypothesis>0.5 else False
  predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
  
  # Launch graph
  with tf.Session() as sess:
      # Initialize TensorFlow variables
      sess.run(tf.global_variables_initializer())
      #printArr(W1.eval())
      #maxA = 0
      #Aarr=[]
      time = 0
      #count = 0
      for step in range(times):
          time+=1
          sess.run(cost, feed_dict={ X: x_data, Y: y_data})
      h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: xl_data, Y: yl_data})
      print("\nAccuracy: ", a, " time : ", time)


values = [4.19, 2.27] # 0.34 ~ 6.12
numbers = [0, 0] # 208013 = 207229 + 784(756 + 28)

for test in range(5):
  x_data=[]
  y_data=[]
  xlist=[]
  ylist=[]
  print("###########################")
  print("############ "+ str(test)+ " ############")
  print("###########################")
  ##########################################################
  ################ data reading & leraning #################
  ##########################################################
  for j in range(5):
    f = open(filename+str(j)+".csv","r")  
    #for title remove
    f.readline() 
    while True:
      tx = []
      ty = []
      line = f.readline()
      if not line: break
      data = line.split(',')
      for i in range(N):
        tx.append(float(data[i]))
      tmp = float(data[N])
      if(tmp > values[0]):
        numbers[0]+=1
        numbers[1]+=1
      elif(tmp > values[1]):
        numbers[1]+=1
      ty.append(tmp)
      if(j==test):
        xlist.append(tx)
        ylist.append(ty)
      else :
        x_data.append(tx)
        y_data.append(ty)  
    f.close()
  

  ##############################################################
  ############################## learning ######################
  ##############################################################  

  times = 1001
  #times = 100  

  range_fal = len(values)
  for i in range(range_fal) :
    print("***************************")
    print("********** "+ str(values[i])+ " ***********")
    print("***************************")
    
    #all method
    x_real = []
    y_real = []
    x_test = []
    y_test = []
    x_data_len=len(x_data)
    data_count=0
    # for value divide
    for j in range(x_data_len):
      if(y_data[j][0]>values[i]):
        y_real.append([1])
        x_real.append(x_data[j])
        data_count+=1
      else :
        y_real.append([0])
        x_real.append(x_data[j])
    x_test_len=len(xlist)
    test_count=0
    # for value divide
    for j in range(x_test_len):
      if(ylist[j][0]>values[i]):
        y_test.append([1])
        x_test.append(xlist[j])
      else :
        y_test.append([0])
        x_test.append(xlist[j])
    #print("learning 1")
    #learning1(x_real, y_real, 0-values[i])
    #print("learning 2")
    learning2(x_real, y_real)
    #print("learning 3")
    #learning3(x_real, y_real, values[i])

