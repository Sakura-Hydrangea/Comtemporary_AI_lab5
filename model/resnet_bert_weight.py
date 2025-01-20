import torch
from torch import nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

class BertResnet(nn.Module):
    """
    - 使用ResNet作为图像特征提取器。
    - 使用BERT作为文本特征提取器。
    - 最终通过加权融合图像和文本特征进行分类。
    """
    
    def __init__(self, args):
        """
        初始化BertResnet模型。
        参数：
        - args: 传递的命令行参数或配置，包括是否使用图像、文本，和ResNet的层数等。
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
            self.resnet.load_state_dict(state_dict)  # 加载权重
            self.img_fc = nn.Linear(1000, 128)  # 将ResNet输出的1000维特征转换为128维

        # 如果使用文本特征
        if self.args.use_text:
            # 加载BERT模型和tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.text_fc = nn.Linear(768, 128)  # 将BERT的输出维度768转换为128维
        
        self.relu = nn.ReLU() 
        self.classifier = nn.Linear(128, 3)  # 最终分类器，将128维特征映射到3类
        self.imgW = nn.Linear(128, 1)  # 图像特征的加权层
        self.textW = nn.Linear(128, 1)  # 文本特征的加权层

    def weight(self, img_feature, text_feature):
        """
        加权融合图像特征和文本特征。
        参数：
        - img_feature: 图像特征，形状为 [batch_size, 128]
        - text_feature: 文本特征，形状为 [batch_size, 128]
        
        返回：
        - feature: 融合后的特征，形状为 [batch_size, 128]
        """
        img_weights = self.imgW(img_feature)  # 计算图像特征的权重 [batch_size, 1]
        text_weights = self.textW(text_feature)  # 计算文本特征的权重 [batch_size, 1]
        
        # 融合特征：按加权系数调整图像和文本特征
        feature = img_weights * img_feature + text_weights * text_feature
        return feature

    def forward(self, text_ids, attention_masks, imgs):
        """
        模型的前向传播函数，处理输入的图像和文本数据。
        参数：
        - text_ids: 输入的文本ID，形状为 [batch_size, seq_len]
        - attention_masks: 输入文本的attention mask，形状为 [batch_size, seq_len]
        - imgs: 输入的图像数据，形状为 [batch_size, 3, H, W]
        
        返回：
        - output: 经过分类器输出的结果，形状为 [batch_size, 3]（三个类别的概率）
        """
        # 如果使用图像特征
        if self.args.use_image:
            img_feature = self.resnet(imgs)  # 通过ResNet提取图像特征 [batch_size, 1000]
            img_feature = self.img_fc(img_feature)  # 将图像特征转换为128维 [batch_size, 128]
        
        # 如果使用文本特征
        if self.args.use_text:
            # 使用BERT提取文本特征 [batch_size, seq_len, 768]
            text_feature = self.bert(text_ids, attention_mask=attention_masks).last_hidden_state[:, 0, :]  
            text_feature = self.text_fc(text_feature)  # 将文本特征转换为128维 [batch_size, 128]
        
        # 如果同时使用图像和文本特征
        if self.args.use_image and self.args.use_text:
            # 加权融合图像和文本特征
            feature = self.weight(img_feature, text_feature)
            output = self.classifier(feature)  # 通过分类器输出 [batch_size, 3]
        # 如果只使用图像特征
        elif self.args.use_image:
            img_feature = self.classifier(img_feature)  # 图像特征直接通过分类器输出
            output = img_feature
        # 如果只使用文本特征
        elif self.args.use_text:
            text_feature = self.classifier(text_feature)  # 文本特征直接通过分类器输出
            output = text_feature
        
        return output
