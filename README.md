训练命令：

UNet

python train_baselines.py --model unet --dataset anfc256 --epochs 50 --batch_size 4
UNet++

python train_baselines.py --model unet++ --dataset anfc256 --epochs 50 --batch_size 4
TransUNet Baseline

python train_unified.py --mode baseline --dataset anfc256 --epochs 50 --batch_size 4
Ours（联合蒸馏）

python train_unified.py --mode ours --dataset anfc256 --lambda_mse 10.0 --lambda_grad 30.0 --epochs 50 --batch_size 4


筛选完成，汇总结果：

子集	原始	保留	剔除
train	2266	2149	117
val	    224	    191	    33
test	501	    452	    49
合计	2991	2792	199
