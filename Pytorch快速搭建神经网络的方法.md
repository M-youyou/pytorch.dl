# Pytorch 快速搭建神经网络的方法 #

Torch提出了类似于Keras那样快速搭建神经网络的方法，构建神经网络时，既可以类似于classification和regression那样的继承class构建神经网络的方式，也可以用Sequential以一种快速的方式构建神经网络这是传统的构建神经网络的方式。

我们用 class 继承了一个 torch 中的神经网络结构， 然后对其进行了修改，我们用net1代表这种方式搭建的神经网络：

	class Net(torch.nn.Module):
	    def __init__(self, n_feature, n_hidden, n_output):
	        super(Net, self).__init__()
	        self.hidden = torch.nn.Linear(n_feature, n_hidden)
	        self.predict = torch.nn.Linear(n_hidden, n_output)
	    def forward(self, x):
	        x = F.relu(self.hidden(x))
	        x = self.predict(x)
	        return x	
	net1 = Net(1, 10, 1)   # 这是我们用这种方式搭建的 net1

不过还有更快的一招, 用一句话就概括了上面所有的内容，这是快速构建神经网络的方式，用net2表示：

	net2 = torch.nn.Sequential(
    	torch.nn.Linear(1, 10),
    	torch.nn.ReLU(),
    	torch.nn.Linear(10, 1)
	)

对比一下两者的结构：
	
	print(net1)
	"""
	Net (
	  (hidden): Linear (1 -> 10)
	  (predict): Linear (10 -> 1)
	)
	"""
	print(net2)
	"""
	Sequential (
	  (0): Linear (1 -> 10)
	  (1): ReLU ()
	  (2): Linear (10 -> 1)
	)
	"""

我们会发现 net2 多显示了一些内容，这是为什么呢？原来他把激励函数也一同纳入进去了，但是 net1 中，激励函数实际上是在 forward() 功能中才被调用的。这也就说明了，相比 net2，net1 的好处就是, 你可以根据你的个人需要更加个性化你自己的前向传播过程, 比如(RNN)。 不过如果你不需要七七八八的过程，相信 net2 这种形式更适合你。[不太理解两个网络在实际应用时，有什么不同？]

这部分的代码在[build_nn_quickly]()