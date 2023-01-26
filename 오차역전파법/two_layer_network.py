#!/usr/bin/env python
# coding: utf-8

# # 오차역전파법을 이용한 신경망

# In[1]:


import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


# In[3]:


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}   # 신경망의 매개변수를 보관하는 딕셔너리 변수
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)   # 1층 가중치
        self.params['b1'] = np.zeros(hidden_size)                                        # 1층 편향
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)  # 2층 가중치
        self.params['b2'] = np.zeros(output_size)                                        # 2층 편향
        
        # Layer 계층 생성
        self.layers = OrderedDict()      # 순서가 있는 딕셔너리
        self.layers['Affine1'] =             Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] =             Affine(self.params['W2'], self.params['b2'])
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):    # 예측, 추혼 함수
        for layer in self.layers.values():      # OrderedDict 딕셔너리의 value 안에서 순전파로 예측
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):   # 손실 함수
        y = self.predict(x)
        
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):      # 정확도 찾기
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        
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
        
        # 순전파
        self.loss(x, t)
        
        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads

