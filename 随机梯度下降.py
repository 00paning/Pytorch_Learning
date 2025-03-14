import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

epoch_list = []
cost_list = []
def forward(x):
    return x*w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2

def gradient(x,y):
    return 2 * (x * w - y)

print('Predict(before training)',4,forward(4))

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        grad = gradient(x,y)
        w = w - 0.01 *grad
        l = loss(x,y)
        cost_list.append(l)
        epoch_list.append(epoch)
    print('Epoch:',epoch,'w=',w,'loss=',l)
print('Predict(after training)',4,forward(4))

plt.plot(epoch_list,cost_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()