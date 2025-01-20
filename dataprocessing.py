from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms  
from PIL import Image 
import torch 
from transformers import BertTokenizer 
from tqdm import tqdm  

class dataset(Dataset):
    def __init__(self, part):
        """
        初始化方法，用于加载不同部分的数据（train、val、test）
        :param phase: 表示数据集的阶段，有 'train'、'val' 或 'test'
        """
        if part == 'train':
            self.index_label = "data/train.txt"
        elif part == 'val':
            self.index_label = "data/val.txt"
        elif part == 'test':
            self.index_label = "data/test_without_label.txt"

        # 读取索引文件，加载文本和标签
        with open(self.index_label) as f:
            f.readline()  
            self.index_label = f.readlines()  
        self.index_label = [i.strip().split(',') for i in self.index_label]  # 切割每行数据，得到 [文件名, 标签] 格式

        # 提取文本文件名、图像文件名和标签
        self.text_files = [i[0] + '.txt' for i in self.index_label] 
        self.img_files = [i[0] + '.jpg' for i in self.index_label]
        
        # 标签转换为数字，'positive' -> 0, 'neutral' -> 1, 'negative' -> 2, 'null' -> -1
        label2idx = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': -1}
        self.labels = [label2idx[i[1]] for i in self.index_label]  # 将标签转换为数字索引
        self.labels = torch.tensor(self.labels)  # 转换为 PyTorch 张量

        # 初始化用于存储文本 ID 和 attention mask 的列表
        self.text_ids_list, self.attention_mask_list = [], []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 使用 BERT 的 Tokenizer 来处理文本
        
        # 对每个文本文件进行处理
        for f in tqdm(self.text_files, desc=f"{part} text tokenize"):  
            with open("data/data/" + f, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().encode("ascii", "ignore").decode()  
                output = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=150)
                text_ids = output['input_ids'].squeeze()  
                attention_mask = output['attention_mask'].squeeze()  # 获取 attention mask
                self.text_ids_list.append(text_ids)  # 保存文本 ID
                self.attention_mask_list.append(attention_mask)  # 保存 attention mask
        
        # 处理图像文件
        self.img_list = []
        for f in tqdm(self.img_files, desc=f"{part} img transform"):  
            img = Image.open("data/data/" + f) 
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # 将图像调整为 256x256 大小，防止有些图像太小
                transforms.RandomCrop(224),  # 随机裁剪为 224x224
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]), 
            ])
            img = transform(img)  
            self.img_list.append(img)  

    def __getitem__(self, index):
        """
        获取数据集中的一个样本
        :param index: 数据索引
        :return: 一个样本的数据，包括文本、图像和标签
        """
        data = (self.text_ids_list[index], self.attention_mask_list[index], self.img_list[index], self.labels[index])
        return data

    def __len__(self):
        return len(self.index_label)

# 创建训练数据加载器
def trainloader(args):    
    trainset = dataset('train', args.modelname)  
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)  
    return trainloader

# 创建验证数据加载器
def valloader(args):
    valset = dataset('val', args.modelname) 
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)  
    return valloader

# 创建测试数据加载器
def testloader(args):
    testset = dataset('test', args.modelname)  
    testloader = DataLoader(testset, batch_size=4, shuffle=False)  
    return testloader
