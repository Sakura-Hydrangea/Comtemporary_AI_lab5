from torch import nn
import torch
from tqdm import tqdm
from dataprocessing import trainloader, valloader

class TrainVal:
    def __init__(self, args):
        """
        初始化训练与验证类的属性。
        :param args: 训练所需的参数，包含学习率、batch size、设备等信息
        """
        self.args = args  
        self.loss = nn.CrossEntropyLoss()  
        self.train_loss_history = []  # 用于存储训练损失的历史记录
        self.train_acc_history = []   # 用于存储训练准确率的历史记录
        self.val_acc_history = []     # 用于存储验证准确率的历史记录
        self.best_acc = -1  
        self.patience = self.args.patience  # 初始化patience，用于早停

    def train(self, model):
        """
        训练模型。
        :param model: 传入需要训练的模型
        """
        self.train_loader = trainloader(self.args)  #
        self.val_loader = valloader(self.args)     
        model.to(self.args.device)  
        
 
        for epoch in range(self.args.epochs):  
            self.train_epoch(model, epoch)  
            self.val(model)  
            if self.patience == 0:  # 如果patience为0，表示准确率没有提升，早停
                print('    early stop.')  
                break  
        self.plot()  

    def train_epoch(self, model, epoch):
        """
        训练一个epoch。
        :param model: 需要训练的模型
        :param epoch: 当前的训练周期
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)  
        model.train()  
        acc = 0  
        
        for batch in tqdm(self.train_loader, desc=f'train epoch {epoch+1}'):
            text_ids, attention_masks, imgs, targets = batch[0].to(self.args.device), batch[1].to(self.args.device), batch[2].to(self.args.device), batch[3].to(self.args.device)
            
            output = model(text_ids, attention_masks, imgs)  
            loss = self.loss(output, targets)  
            loss.backward()  
            optimizer.step()  
            optimizer.zero_grad()  
            
            # 累加每个batch的准确率
            acc += self.getacc(output, targets) / len(self.train_loader)
        
        # 输出当前epoch的损失和准确率
        print('    train loss: {}'.format(loss.item()))
        print('    train acc: {}'.format(acc))

        # 将当前epoch的损失和准确率添加到历史记录中
        self.train_loss_history.append(loss.item())
        self.train_acc_history.append(acc)

    def val(self, model):
        """
        验证模型性能。
        :param model: 需要验证的模型
        """
        print('    val...')
        model.eval()  
        acc = 0  
        
        # 遍历验证数据集中的每个batch
        for batch in self.val_loader:
            text_ids, attention_masks, imgs, targets = batch[0].to(self.args.device), batch[1].to(self.args.device), batch[2].to(self.args.device), batch[3].to(self.args.device)
            
            output = model(text_ids, attention_masks, imgs) 
            acc += self.getacc(output, targets) / len(self.val_loader)  
        
        # 输出当前验证的准确率
        print('    val acc: {}'.format(acc))
        self.val_acc_history.append(acc)  # 将验证准确率添加到历史记录中
        
        # 如果当前验证准确率为最佳，则保存模型
        if acc > self.best_acc:
            self.best_acc = acc  # 更新最佳准确率
            torch.save(model.state_dict(), f'trained_bert_resnet/{self.args.version}.pth')  # 保存模型
            print('    model saved: trained_bert_resnet/{}.pth'.format(self.args.version))  # 输出保存的模型路径
            self.patience = self.args.patience  
        else:
            self.patience -= 1  

    def getacc(self, output, targets):
        """
        计算准确率。
        :param output: 模型的输出结果
        :param targets: 真实标签
        :return: 计算出的准确率
        """
        pred = output.argmax(dim=1)  
        acc = (pred == targets).sum().item() / len(pred) 
        return acc

    def plot(self):
        """
        绘制训练和验证过程中的准确率曲线。
        """
        import matplotlib.pyplot as plt
        plt.title('Accuracy')  
        plt.plot(self.train_acc_history, label='train')  
        plt.plot(self.val_acc_history, label='val')  
        plt.legend()  
        plt.show()  
