
import argparse
from train import TrainVal
from test import Test
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":

    # 初始化 ArgumentParser
    parser = argparse.ArgumentParser()

    # 添加版本参数
    parser.add_argument('--version', type=str, default='bert_resnet50')

    # 添加其他参数，并为它们设置默认值
    parser.add_argument('--modelname', type=str, default='resnet_bert_concat')
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--resnet', type=int, default=50)

    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--use_image', type=int, default=1)

    parser.add_argument('--train_or_test', type=str, default="train")
    args = parser.parse_args()
    
    train_val = TrainVal(args)
    test = Test(args)
    modelname = args.modelname.lower()
    if modelname == 'resnet_bert_concat':
        from model.resnet_bert_concat import BertResnet
        model = BertResnet(args).to(args.device)
    elif modelname == 'resnet_bert_weight':
        from model.resnet_bert_weight import BertResnet
        model = BertResnet(args).to(args.device)
    if args.train_or_test == "train":
        train_val.train(model)   # train, val
    elif args.train_or_test == "test":
        Test.test(model, f'trained_bert_resnet\resnet50_concat.pth') #注意！在这里放入你训练好的.pth权重文件