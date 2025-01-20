#!/bin/bash

# 超参数配置
learning_rates=("0.000001" "0.00001" "0.0001")
batch_sizes=("16" "32" "64")
# resnets=("18" "50")

train_or_test="train"

# 循环遍历不同的超参数组合
for lr in "${learning_rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for resnet in "${resnets[@]}"; do

            # 打印当前训练配置
            echo "Running with: lr=$lr, batch_size=$batch_size, resnet=$resnet"

            # 运行训练命令
            python main.py \
                --lr $lr \
                --batch_size $batch_size \
                --resnet $resnet \
                --train_or_test $train_or_test

        done
    done
done

echo "自动调参完成！"
