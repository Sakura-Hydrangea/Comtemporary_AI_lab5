## 人工智能大作业 

本次实验的任务是给定配对的文本和图像，预测对应的情感标签。是一个三分类任务：positive, neutral, negative。
请注意，data文件夹下的原始数据我在发送邮件时删掉了（因为太大了），因此复现需要将原始数据重新放入data目录中

## 依赖

运行如下命令安装依赖：

```
pip install -r requirements.txt
```

## 代码结构

```
├─data/
│ ├─data/           # 原始数据
│ │  ├─1.jpg        # 图像数据
│ │  ├─1.txt        # 文本数据
│ │  └─...jpg
│ ├─test_without_label.txt     # 测试集
│ ├─train.txt                  # 训练集
│ └─val.txt                    # 验证集
├─pretrained/			#保存ResNet的预训练模型
├─split_dataset.py       # 划分训练集和验证集到data目录下
├─dataprocessing.py       # 数据预处理
├─main.py          # 运行入口
├─model/           # 不同的模型实现
│  ├─resnet_bert_weight.py   # 加权融合bert和resnet提取文本和图片特征
│  └─resnet_bert_concat.py   # 拼接融合bert和resnet提取文本和图片特征
├─trained_bert_resnet      #保存训练好的.pth权重
├─train.py       # 训练和验证
├─test.py       # 测试
├─hyperparameter.sh      # 超参数调优脚本
└─result.txt         # 预测结果文件
```

## 运行代码

首先要下载ResNet18和ResNet50的预训练模型，并将其放入pretrained_model目录下，下载链接如下：

```
https://download.pytorch.org/models/resnet18-5c106cde.pth
https://download.pytorch.org/models/resnet50-19c8e357.pth
```

模型接受以下重要的参数：

+ 学习率
+ batch size
+ epochs
+ resnet （选择resnet18或50）
+ use_text 是否使用文本，如果用则为1，不用为0
+ use_image 是否使用图像，如果用则为1，不用为0
+ train or test 选择训练还是测试
+ modelname 选择使用什么模型
+ version 版本控制，可以自己定义version的输入名称是什么，便于区分不同次训练的结果

```
parser.add_argument('--version', type=str, default='bert_resnet50')
parser.add_argument('--modelname', type=str, default='bert_resnet_concat')
parser.add_argument('--lr', type=float, default=0.000001)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda')

parser.add_argument('--resnet', type=int, default=50)

parser.add_argument('--use_text', type=int, default=1)
parser.add_argument('--use_image', type=int, default=1)

parser.add_argument('--train_or_test', type=str, default="train")
```

运行示例：

```
python main.py --modelname resnet_bert_concat --version resnet50_concat --resnet 50
```

