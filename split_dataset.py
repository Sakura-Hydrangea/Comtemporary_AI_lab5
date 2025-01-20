import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_train_val(val_ratio=0.125, random_state=42):
    """
    从 train.txt 中划分出验证集
    
    参数:
    - val_ratio: 验证集比例 
    - random_state: 随机种子
    """
    # 读取原始训练数据
    df = pd.read_csv('data/train.txt')
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(
        df,
        test_size=val_ratio,  
        random_state=random_state,
        stratify=df['tag']  # 确保标签分布一致
    )
    
    # 保存划分后的数据集
    train_data.to_csv('data/train.txt', index=False)
    val_data.to_csv('data/val.txt', index=False)
    
    # 打印数据集统计信息
    print(f"原始数据集总大小: {len(df)}")
    print(f"训练集大小: {len(train_data)} ")
    print(f"验证集大小: {len(val_data)} ")
    
    # 打印每个数据集的标签分布
    print("\n标签分布:")
    print("训练集:", train_data['tag'].value_counts().to_dict())
    print("验证集:", val_data['tag'].value_counts().to_dict())

if __name__ == "__main__":
    split_train_val(
        val_ratio=0.125,
        random_state=42
    )