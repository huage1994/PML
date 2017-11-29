import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),0).type(torch.LongTensor)

x,y = Variable(x),Variable(y)


# plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(), s= 100,lw=0)
# plt.show()


'''
z = torch.FloatTensor([[1,2],[3,4]])
print(z)
print(torch.cat((z,z),1))
'''


class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_out):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.out = torch.nn.Linear(n_hidden,n_out)

    def forward(self, x):
        x = self.hidden(x)
        x = self.out(F.relu(x))
        return x

net = Net(2,10,2)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.02)


plt.ion()
plt.show()

for epoch in range(300):
    out = net(x)
    loss = loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%5 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200
        plt.text(0.5, -4,'%d ,Accuracy=%.2f' % (epoch,accuracy), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
