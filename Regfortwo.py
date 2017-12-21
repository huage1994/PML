import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

EPOCH = 400

x = torch.linspace(0,2,100)
x = torch.unsqueeze(x,1)
# print(x.view(1,-1,4))
print(x)
x1 = torch.randn([100,1])
y = x*x1 + 1 + torch.randn([100,1]) *0.2
print(y)

x = torch.cat((x,x1),1)

y = Variable(y)
x = Variable(x)

net1 = nn.Sequential(
    nn.Linear(2,30),
    nn.ReLU(),
    nn.Linear(30,1),
)

loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(net1.parameters(),lr=0.03,momentum=0.8)


plt.ion()
plt.show()
plt.ylim((0,6))

for epoch in range(EPOCH):
    prediction = net1(x)
    loss = loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%5 == 0:
        plt.cla()
        plt.scatter(x.data[:,0].numpy(),y.data.numpy())
        plt.plot(x.data[:,0].numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(1.5,0, '%d loss=%.4f'% (epoch,loss.data[0]),fontdict={'size':20,'color':'red'} )
        plt.pause(0.1)

plt.pause(100)
plt.ioff()
plt.close()