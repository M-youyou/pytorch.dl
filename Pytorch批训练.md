# Pytorch批训练 #
Torch提供了一种帮助整理数据结构的功能，叫做DataLoader，能够使用它包装自己的数据，进行批训练。除了DataLoader之外，批训练也有很多其他的方式，如：

- 批训练的方式之一

Dataloader是torch封装好的工具，需要用户将自己的数据转化为Tensor形式，然后再放进这个包装器中，使用DataLoader的好处是能够帮助用户有效迭代数据，举例如下：

	import torch
	import torch.utils.data as Data
	torch.manual_seed(1)    # reproducible
	
	BATCH_SIZE = 5      # 批训练的数据个数
	
	x = torch.linspace(1, 10, 10)       # x data (torch tensor)
	y = torch.linspace(10, 1, 10)       # y data (torch tensor)
	
	# 先转换成 torch 能识别的 Dataset
	torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
	
	# 把 dataset 放入 DataLoader
	loader = Data.DataLoader(
	    dataset=torch_dataset,      # torch TensorDataset format
	    batch_size=BATCH_SIZE,      # mini batch size
	    shuffle=True,               # 要不要打乱数据 (打乱比较好)
	    num_workers=2,              # 多线程来读数据
	)
	
	for epoch in range(3):   # 训练所有!整套!数据 3 次
	    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
	        # 假设这里就是你训练的地方...
	
	        # 打出来一些数据
	        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
	              batch_x.numpy(), '| batch y: ', batch_y.numpy())
	
	"""
	Epoch:  0 | Step:  0 | batch x:  [ 6.  7.  2.  3.  1.] | batch y:  [  5.   4.   9.   8.  10.]
	Epoch:  0 | Step:  1 | batch x:  [  9.  10.   4.   8.   5.] | batch y:  [ 2.  1.  7.  3.  6.]
	Epoch:  1 | Step:  0 | batch x:  [  3.   4.   2.   9.  10.] | batch y:  [ 8.  7.  9.  2.  1.]
	Epoch:  1 | Step:  1 | batch x:  [ 1.  7.  8.  5.  6.] | batch y:  [ 10.   4.   3.   6.   5.]
	Epoch:  2 | Step:  0 | batch x:  [ 3.  9.  2.  6.  7.] | batch y:  [ 8.  2.  9.  5.  4.]
	Epoch:  2 | Step:  1 | batch x:  [ 10.   4.   8.   1.   5.] | batch y:  [  1.   7.   3.  10.   6.]
	"""

以上，每一步都导出了5个数据，每个epoch的导出数据都是先打乱了再导出。

另外，我们还可以改变`BATCH_SIZE`的大小，如`BATCH_SIZE=8`，这样每个step便会导出`8`个数据：

	BATCH_SIZE = 8      # 批训练的数据个数
	
	...
	
	for ...:
	    for ...:
	        ...
	        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
	              batch_x.numpy(), '| batch y: ', batch_y.numpy())
	"""
	Epoch:  0 | Step:  0 | batch x:  [  6.   7.   2.   3.   1.   9.  10.   4.] | batch y:  [  5.   4.   9.   8.  10.   2.   1.   7.]
	Epoch:  0 | Step:  1 | batch x:  [ 8.  5.] | batch y:  [ 3.  6.]
	Epoch:  1 | Step:  0 | batch x:  [  3.   4.   2.   9.  10.   1.   7.   8.] | batch y:  [  8.   7.   9.   2.   1.  10.   4.   3.]
	Epoch:  1 | Step:  1 | batch x:  [ 5.  6.] | batch y:  [ 6.  5.]
	Epoch:  2 | Step:  0 | batch x:  [  3.   9.   2.   6.   7.  10.   4.   8.] | batch y:  [ 8.  2.  9.  5.  4.  1.  7.  3.]
	Epoch:  2 | Step:  1 | batch x:  [ 1.  5.] | batch y:  [ 10.   6.]
	"""

以上，训练代码在[batch_train.py]()
