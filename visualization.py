import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


x = np.linspace(-5,5,200)
print (x)


# visulization
'''
plt.figure()

plt.subplot(211)
plt.plot([0,1],[0,1])

plt.subplot(234)
plt.plot([0,1],[0,2])

plt.subplot(235)
plt.plot([0,1],[0,3])

plt.subplot(236)
plt.plot([0,1],[0,4])

plt.show()
plt.close()
'''


plt.figure()

plt.subplot(311)
plt.plot([0,1],[0,1])
plt.show()