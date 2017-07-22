import numpy as np

'''
#text example
current_level | raining | next_level
       1      |  0.0    |    1.0
       1      |  0.1    |    1.1
       2      |  0.0    |    2.0
       2      |  0.1    |    2.1
'''

X = np.array([[1, 0], [1, 0.1], [2, 0], [2, 0.1]]).T
Y = np.array([[1.1, 1, 2, 2.1]])
W = np.random.random((1, 2))

learning_rate = 0.1
epoch = 100


print('Weight Before Traning: (' + str(W[0]) + ')')
for _ in range(epoch) :
	dW = -1*(Y-W.dot(X)).dot(X.T)*1
	W -= learning_rate*dW

print('Weight After Traning: (' + str(W[0]) + ')')
