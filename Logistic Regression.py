import torch.nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[0.0],[0.0],[1.0]])

loss_list = []

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    loss_list.append(loss.item())
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 绘制损失曲线
plt.figure(figsize=(8, 5))
plt.plot(loss_list, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

x = np.linspace(0,10,200)
x_t = torch.tensor(x, dtype=torch.float32).view((200,1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('hours')
plt.ylabel('probability of pass')
plt.grid()
plt.show()