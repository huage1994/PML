import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

'''
# axis basic plot 

x = np.linspace(-1,1,100)
y = x**2 +1


plt.figure(figsize=(5,5))
l1, = plt.plot(x,y-1,color='yellow',linewidth=1.0,label='down')
l2, = plt.plot(x,y,color='red',linewidth=1.0)
l3, = plt.plot(x,y+1,linewidth=1.0,label='up')
plt.ylim((0,3))
plt.yticks(np.linspace(0,1,5))
plt.legend(handles=[l1,l2,l3,], labels =['1','2','eat'], loc='best')

plt.figure(figsize=(6,6))
plt.plot(x,y+1,linewidth=5.0)
plt.xlim((-2,2))
plt.ylim((0,2))
plt.xlabel('I am boy')
plt.ylabel('I am girl')
plt.xticks(np.linspace(-2,2,5))
plt.yticks(np.linspace(0,6,7))




# plt.yticks(np.linspace(0,6,3),[r'$bad\ \alpha$','$test$','good'])

# plt.text(-2,3,s=' wo hsi ni baba',fontdict={'size':16,'color':'r'})
# plt.scatter(1,3,s=100,c='red')
# plt.plot([2,1],[2,3],'k--',lw=2.5)
x0 = 1
y0 = x0**2+2
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

###axis 2
ax = plt.gca()   #
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',1))  # outward, axes(% percent).
ax.spines['left'].set_position(('data',0))

for ticklabel in ax.get_xticklabels()+ ax.get_yticklabels():
    ticklabel.set_fontsize(12)
    ticklabel.set_bbox(dict(facecolor='red',edgecolor='None',alpha=0.7))

plt.show()
'''


'''
# plt.plt([1,2,3,4])          list index as x automatically

list = [x*x for x in range(0,20,2)]
plt.plot(list)
print(list)
plt.xticks(np.linspace(0,19,10))
plt.show()

# substitute
plt.plot([x*x for x in range(20)])
plt.show()
'''

'''
#scatter beautiful

X = np.random.normal(0,1,1000)
Y = np.random.normal(0,1,1000)
T = np.arctan2(Y,X)
plt.scatter(X,Y,c=T,alpha=0.5,s=75)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.xticks(())
plt.yticks(())
plt.show()
'''

'''
# Bar

X = np.arange(12)
Y = (1 - X/12) * np.random.uniform(0.5,1,12)
print(X/float(12),Y)
plt.ylim(np.min(Y),np.max(Y)+0.3)
plt.bar(X,+Y,facecolor='#ff9999',edgecolor='black')
for x,y in zip(X,Y):
    plt.text(x,y+0.01,'%.2f' %y,ha='center',va='bottom')

plt.show()
'''
"""
X = np.arange(12)
print(np.exp(1))
"""

f = plt.figure()
ax = Axes3D(f)
X = np.arange(-4,4,0.25)
Y = np.arange(-4,4,0.25)


X,Y = np.meshgrid(X,Y)
R =X*Y
print(R.shape)                  # R is 32 *32


ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap='rainbow')

plt.show()

