import numpy as np
import matplotlib.pyplot as plt
import torch
import struct


"""
x = np.linspace(-5,5,200)
x = torch.from_numpy(x)
x = x.type(torch.FloatTensor)
print (torch.unsqueeze(x,1))

"""

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
'''

'''
#show picture in mnist

filename = './mnist/raw/train-images-idx3-ubyte'
binfile = open(filename, 'rb')
buf = binfile.read()

index = 0
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')

im = struct.unpack_from('>784B', buf, index)
index += struct.calcsize('>784B')

im = np.array(im)
im = im.reshape(28, 28)

fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(im, cmap='gray')
plt.show()



data = torch.randn([4,2])
print(data)

print(torch.max(data,1)[1])

print(torch.cat((data,data),0))
'''


# test1

x = np.linspace(-1,1,100)
y = x**2 +1


plt.figure(figsize=(5,5))
plt.plot(x,y-1,color='yellow',linewidth=1.0)
plt.plot(x,y,color='red',linewidth=1.0)
plt.plot(x,y+1,linewidth=1.0)

plt.figure(figsize=(6,6))
plt.plot(x,y+1,linewidth=10.0)
plt.xlim((-2,2))
plt.ylim((0,6))
plt.xlabel('I am boy')
plt.ylabel('I am girl')
plt.xticks(np.linspace(-2,2,5))
plt.yticks(np.linspace(0,6,3),[r'$bad\ \alpha$','$test$','good'])

###axis 2
ax = plt.gca()   #
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',1))  # outward, axes(% percent).
ax.spines['left'].set_position(('data',0))

plt.show()
