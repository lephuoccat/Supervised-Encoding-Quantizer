#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:49:17 2019

@author: catle
"""

import numpy as np
import matplotlib.pyplot as plt

# data on fashion-mnist
k_range = np.linspace(10,100,10)
accuracy = np.array([71.95, 88.89, 90.35, 90.45, 90.99, 90.83, 91.04, 90.95, 91.14, 91.14])

plt.plot(k_range, accuracy)
plt.xlabel('Number of clusters (K)')
plt.ylabel('Accuracy (%)')
plt.title('Performance of deep semi-generative learning on fashion-MNIST')
plt.axis([10,100,70,95])
plt.grid(True)
#plt.show()
 
plt.savefig('fashion-mnist-acc.png')
