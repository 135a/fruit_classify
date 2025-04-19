# fruit_classify
这是一个python脚本，用于根据水果图片对水果进行分类。

数据集下载链接:https://www.kaggle.com/datasets/shreyapmaher/fruits-dataset-images

核心算法：卷积神经网络

![image](https://github.com/user-attachments/assets/0af3ad39-21f7-419b-bcde-2e2ad584c3df)

可见，我们设置了一个两个卷积层，两个池化层，共20082505个参数

同时，为了防止过拟合，我们进行了随机正则化，并采用了早停法

具体参数需在反复训练尝试中逐渐解决，正如人们所调侃的：人工智能，有多少人工，就有多少智能

![01b214b3e92a008cb805e645ec22acbb](https://github.com/user-attachments/assets/42cc62ee-83d1-4e7f-a12e-78f2f6045082)


经反复训练，模型的最大准确率约为70%
