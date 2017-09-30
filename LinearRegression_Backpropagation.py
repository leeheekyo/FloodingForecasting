# Lab 9 XOR
import tensorflow as tf
import numpy as np
import random
from datetime import datetime

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

filename="data_final"

N = 6

times = 0

x_test = []
y_test = []

#select dangerous and warning value
values = [4.19, 2.27] # 0.34 ~ 6.12

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
  return(sumval1, total1, sumval0, total0, sumval1+sumval0 ,x_test_len)

def calc2(mat1, sub1, mat2, sub2):
  sumval = 0
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
  print(sumval, x_test_len)

def calc3(mat1, sub1, mat2, sub2, mat3, sub3):
  sumval = 0
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
  print(sumval, x_test_len)

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
        #if(j!=j_length-1):
        #  pval+=","
        pval+=","
      #pval=pval[0:-1]
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
def learning1(x_data, y_data, c_W1, c_b1):
  x_data = np.array(x_data, dtype=np.float32)
  y_data = np.array(y_data, dtype=np.float32)
  
  X = tf.placeholder(tf.float32, [None, N])
  Y = tf.placeholder(tf.float32, [None, 1])
  
 
  W1 = tf.Variable(np.float32(c_W1), name='weight1')
  b1 = tf.Variable(np.float32(c_b1), name='bias1')
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
      sess.run(cost, feed_dict={ X: x_data, Y: y_data})

      for step in range(times):
          sess.run(cost, feed_dict={ X: x_data, Y: y_data})
          sess.run([W1])
          if step%100 == 0 :
            h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
            print("\nAccuracy: ", a, " times : ", step)
      return(W1.eval(), b1.eval())
  

numbers = [0, 0] # 208013 = 207229 + 784(756 + 28)

for test in range(N-1):
  x_data=[]
  x_data1=[]
  x_data2=[]
  y_data=[]
  y_data1=[]
  y_data2=[]
  x_test=[]
  x_test1=[]
  x_test2=[]
  y_test=[]
  y_test1=[]
  y_test2=[]
  print("###########################")
  print("########### "+ str(test)+ " #############")
  print("###########################")

  ##########################################################
  ################ data reading & leraning #################
  ##########################################################
  for j in range(5):
    f = open("4"+filename+str(j)+".csv","r")
    f2 = open("2"+filename+str(j)+".csv","r")  
    #for title remove
    f.readline() 
    f2.readline() 
    while True:
      tx = []
      tx2 = []
      ty = []
      ty2 = []
      line = f.readline()
      line2 = f2.readline()
      if not line: break
      data = line.split(',')
      data2 = line2.split(',')
      for i in range(N):
        tx.append(float(data[i]))
        tx2.append(float(data2[i]))
      data[N]=data[N].rstrip()
      data2[N]=data2[N].rstrip()
      tmp = float(data[N])
      tmp2 = float(data2[N])
      ty.append(tmp)
      ty2.append(tmp2)
      if(j==test):
        x_test1.append(tx)
        x_test2.append(tx2)
        y_test1.append(ty)
        y_test2.append(ty2)
      else :
        x_data1.append(tx)
        x_data2.append(tx2)
        y_data1.append(ty)  
        y_data2.append(ty2)  
    f.close()
  

  ##############################################################
  ############################## learning ######################
  ##############################################################  

  times = 1001

  range_fal = len(values)

  for i in range(range_fal) :
    B_Ctmp=0
    if(i==0):
      x_data = x_data1
      x_test = x_test1
      y_data = y_data1
      y_test = y_test1
    else:
      x_data = x_data2
      x_test = x_test2
      y_data = y_data2
      y_test = y_test2
    print("***************************")
    print("********** "+ str(values[i])+ " ***********")
    print("***************************")

    print("learning 1")
    B_W1 = []
    B_b1 = []
    B_Ctmp1 =0
    B_Ctmp_total1 =0
    B_Ctmp0 =0
    B_Ctmp_total0 =0
    B_Ctmp =0
    B_Ctmp_total =0

    for ii in range(1):
      W1 = []
      b1 = [0]
      ttt=0
      for j in range( N):
        t_W1=np.random.rand(1)
        tt_W1=[]
        tt_W1.append(t_W1[0])
        W1.append(tt_W1)
      for j in range(N):
        ttt+=W1[j][0]
      for j in range(N):
        W1[j][0]/=ttt
      if(i==0):
        b1=[-0.67]
      else:
        b1=[-0.51]
      
      W1, b1 = learning1(x_data, y_data, W1, b1)
      tmpC1, totalC1, tmpC0, totalC0, tmpC, totalC = calc1(W1,b1)

      #find best fit...
      if(B_Ctmp1 <= tmpC1 and B_Ctmp<=tmpC ):
        B_Ctmp_total1 = totalC1
        B_Ctmp1=tmpC1
        B_Ctmp_total0 = totalC0
        B_Ctmp0=tmpC0
        B_Ctmp_total = totalC
        B_Ctmp=tmpC
        B_W1=W1
        B_b1=b1
    print("sumval1 : " + str(B_Ctmp1) +", total1 : " + str(B_Ctmp_total1))
    print("sumval0 : " + str(B_Ctmp0) +", total0 : " + str(B_Ctmp_total0))
    print("sumval  : " + str(B_Ctmp) +", total   : " + str(B_Ctmp_total))
    

