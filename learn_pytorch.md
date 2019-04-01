# Pytorch教程 #

----------
- 参考莫烦的Pytorch教程

----------

## 第一个使用Pytorch构建神经网络的实例 ##
### 建立数据集 ###
我们创建一些假数据来模拟真实的情况。 比如一个一元二次函数: $ y = a * x^2 + b $， 我们给 $y$ 数据加上一点噪声来更加真实的展示它。

(```)

	import torch
	import matplotlib.pyplot as plt
	
	x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
	y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
	
	# 画图
	plt.scatter(x.data.numpy(), y.data.numpy())
	plt.show()

(```)
### 建立神经网络 ###
### 训练网络 ###
### 可视化训练过程 ###