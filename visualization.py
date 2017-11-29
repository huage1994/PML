import numpy as np
import matplotlib.pyplot as plt
import torch


x = np.linspace(-5,5,200)
x = torch.from_numpy(x)
x = x.type(torch.FloatTensor)
print (torch.unsqueeze(x,1))


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

x = np.linspace(-1,1,50) *3
y1 = x*2 +1

y1[25] = 0

y2 = x**2

y3 = torch.randn(100,1)
y3 = torch.cat((y3,y3+1),1)
print(y3)

plt.figure()

plt.plot(x,y1)
plt.plot(x,y2,color='yellow',linewidth=1.0)    # ,linestyle='--'
plt.show()