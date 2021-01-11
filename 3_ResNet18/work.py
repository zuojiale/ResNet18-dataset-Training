#-*-coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from utils import plot_image, plot_curve, one_hot
from matplotlib import pyplot as plt
import cnn
from cnn import ResNet18
import MyDataloader
from MyDataloader import MyDataset


# 定义一些超参数
batch_size = 10         # 一次喂入的图片数
learning_rate = 0.01

num_epoches = 20

# 数据预处理。transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）
# transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
# transforms.Compose()函数则是将各种预处理的操作组合到了一起

# 1.数据集准备
# 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
#root = "F:/Rgb_Cnn/"
root1 = '/home/zjl/HRG_hand/1_OpencvTof/samples/opencv/Test_Pics/'
root2 = '/home/zjl/HRG_hand/1_OpencvTof/samples/opencv/Test_Pics/'
trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
train_data = MyDataset(root=root1, datatxt='Train_list.txt', transform=trans)
test_data = MyDataset(root=root2, datatxt='Train_list.txt', transform=trans)

# 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


# 随机选取测试集样片看看
x, y = next(iter(train_loader))
print(x.shape, y.shape)
plot_image(x, y, 'image sample')

# 选择模型
model = ResNet18()
# model = net.Activation_Net(28 * 28, 300, 100, 10)
# model = net.Batch_Net(28 * 28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
epoch = 0
train_loss = []
for i in range(num_epoches):
    for data in train_loader:
        img, label = data
        # img = img.view(img.size(0), -1)
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        epoch += 1
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

plot_curve(train_loss, 'blue', 'train_loss')

###模型保存

torch.save(model, 'HRG_python.pth')


# 模型评估
model.eval()
eval_loss = 0
eval_acc = 0
test_acc = []
for data in test_loader:
    img, label = data
    # img = img.view(img.size(0), -1)
    img = Variable(img)
    label = Variable(label)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data.item()*label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)),eval_acc / (len(test_data))))
    test_acc.append(eval_acc / (len(test_data)))

plot_curve(test_acc, 'red', 'test_acc')

###模型保存成c++可以读取的格式
example = torch.rand(1, 3, 32, 32)

if torch.cuda.is_available():
    example = example.cuda()
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("HRG_C++.pt")
print("model save success!")
