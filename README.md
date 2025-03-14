# Pytorch_Learning
  记录学习pytorch的相关经历，更新ing
## 1.Linear Moedel
- 由于是初级阶段，这里采用穷举法对w进行取值，观察图像看不同w对应的mse的变化情况  
## 2.梯度下降法  
- 下面是关于梯度下降法的初始公式，具体的推导可以相应的求偏导即可  
- 首先gradient是每次训练时的cost对w求偏导,alpha表示学习率，更新的w等于每次的gradient乘以学习率  
$$w = w - \alpha \frac {dcost}{dw}$$  
