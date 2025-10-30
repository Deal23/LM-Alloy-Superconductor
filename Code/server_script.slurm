#!/bin/bash
#SBATCH -J LM_Alloy #指定作业名称
#SBATCH -p normal #指定分区名称，根据手册，使用正式账号可以提交到normal分区
#SBATCH -N 1 #指定节点数量
#SBATCH -n 30 #指定每个节点的核心数量
#SBATCH --mem=100G  # 指定任务运行内存
#SBATCH -t 24:00:00 # 设置作业运行时间，不超过一天
#SBATCH -o transformer_train.o%j #指定正确输出文件名称
#SBATCH -e transformer_train.e%j #指定报错信息输出文件名称


# 作业结束后，输出结束时间
echo "Training    began  at $(date)"
# 执行训练脚本
python Path_to_your_train.py
# 作业结束后，输出结束时间
echo "Training completed at $(date)"