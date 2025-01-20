import torch
from torch import nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel

class BertResnet(nn.Module):
    
    def __init__(self, args):
        """
        初始化BertResnet模型。
        参数：
        - args: 传递的命令行参数或配置，包括是否使用图像、文本，以及ResNet层数等。
        """
        super(BertResnet, self).__init__()
        self.args = args
        
        # 如果使用图像特征
        if self.args.use_image:
            # 根据配置选择ResNet层数
            if self.args.resnet == 18:
                self.resnet = models.resnet18(pretrained=False)  
                state_dict = torch.load('pretrained_model/resnet18-5c106cde.pth')  
            elif self.args.resnet == 50:
                self.resnet = models.resnet50(pretrained=False) 
                state_dict = torch.load('pretrained_model/resnet50-19c8e357.pth')  
            self.resnet.load_state_dict(state_dict)  # 加载ResNet的预训练权重
            self.img_fc = nn.Linear(1000, 128)  

        # 如果使用文本特征
        if self.args.use_text:
            # 加载BERT模型和tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.text_fc = nn.Linear(768, 128)  # 文本特征的全连接层，将BERT的768维输出映射到128维
        
        # 根据是否同时使用图像和文本，选择不同的融合方式
        if self.args.use_image and self.args.use_text:
            # 如果使用图像和文本，将二者的特征拼接后通过一个全连接层融合
            self.fusion = nn.Linear(128 + 128, 3)
        else:
            # 如果只使用图像或文本，则单独对其进行分类
            self.fc = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, text_ids, attention_masks, imgs):
        """
        前向传播函数，处理输入的文本和图像数据。
        参数：
        - text_ids: 输入的文本ID，形状为 [batch_size, seq_len]
        - attention_masks: 输入文本的attention mask，形状为 [batch_size, seq_len]
        - imgs: 输入的图像数据，形状为 [batch_size, 3, H, W]
        
        返回：
        - output: 经过分类器输出的结果，形状为 [batch_size, 3]（三个类别的概率）
        """
        
        # 如果使用图像特征
        if self.args.use_image:
            img_feature = self.resnet(imgs)  # 通过ResNet提取图像特征，形状为 [batch_size, 1000]
            img_feature = self.img_fc(img_feature)  # 将图像特征从1000维映射到128维，形状为 [batch_size, 128]
            img_feature = self.relu(img_feature)  # 使用ReLU激活函数，形状为 [batch_size, 128]
        
        # 如果使用文本特征
        if self.args.use_text:
            # 使用BERT提取文本特征，形状为 [batch_size, seq_len, 768]
            text_feature = self.bert(text_ids, attention_mask=attention_masks).last_hidden_state[:, 0, :]
            text_feature = self.text_fc(text_feature)  # 将文本特征从768维映射到128维，形状为 [batch_size, 128]
            text_feature = self.relu(text_feature)  # 使用ReLU激活函数，形状为 [batch_size, 128]
        
        # 如果同时使用图像和文本特征
        if self.args.use_image and self.args.use_text:
            feature = torch.cat([img_feature, text_feature], dim=1)  # 将图像和文本特征拼接，形状为 [batch_size, 128 + 128]
            output = self.fusion(feature)  # 通过融合层得到最终输出，形状为 [batch_size, 3]
        # 如果只使用图像特征
        elif self.args.use_image:
            img_feature = self.fc(img_feature)  # 通过全连接层对图像特征进行分类，形状为 [batch_size, 3]
            output = img_feature
        # 如果只使用文本特征
        elif self.args.use_text:
            text_feature = self.fc(text_feature)  # 通过全连接层对文本特征进行分类，形状为 [batch_size, 3]
            output = text_feature

        return output
