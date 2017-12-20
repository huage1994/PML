import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-1,1,100)
print(x)
x = torch.unsqueeze(x,1)
print(x)

y = x **2 + torch.rand([100,1]) * 0.2

x,y = Variable(x),Variable(y)
# plt.scatter(x.numpy(),y.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_out):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.out = torch.nn.Linear(n_hidden,n_out)

    def forward(self, x):
        x = self.hidden(x)
        x = self.out(F.relu(x))
        return x


def save():
    net = Net(1,10,1)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.3)


    plt.ion()
    plt.show()

    for epoch in range(300):
        prediction = net(x)
        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%5 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(),y.data.numpy())
            plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
            plt.text(0.44,0, '%d loss=%.4f'% (epoch,loss.data[0]),fontdict={'size':20,'color':'red'} )
            plt.pause(0.1)

    plt.ioff()
    plt.close()
    torch.save(net.state_dict(),'./dataset/net_state.pkl')


def load():
    net = torch.load('./dataset/net.pkl')
    predict = net(x)
    plt.figure()
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw=5)

    plt.show()


def load_fromstate():
    net = Net(1, 10, 1)
    net.load_state_dict(torch.load('./dataset/net_state.pkl'))
    predict = net(x)
    plt.figure()
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw=5)

    plt.show()


if __name__ == '__main__':
    load_fromstate()




