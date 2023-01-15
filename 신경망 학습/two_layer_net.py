#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient


# In[2]:


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}   # 신경망의 매개변수를 보관하는 딕셔너리 변수
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)   # 1층 가중치
        self.params['b1'] = np.zeros(hidden_size)                                        # 1층 편향
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)  # 2층 가중치
        self.params['b2'] = np.zeros(output_size)                                        # 2층 편향

    def predict(self, x):    # 예측, 추혼 함수
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):   # 손실 함수
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):      # 정확도 찾기
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):         # 가중치 매개변수의 기울기
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}        # 기울기 보관 변수
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):    # numerical_gradient 개선법 - 기울기 고속으로 구하기
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

