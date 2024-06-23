#!/bin/sh
PARTITION=Segmentation


GPU_ID=1,6 # GPU ID
dataset=LoveDA # iSAID/LoveDA

exp_name=split0 # FILE_NAME

arch=MGANet

net=resnet50 # vgg resnet50

model=model/${arch}.py
Cocontrast=model/ProtoContrastModule.py
exp_dir=exp/MGANet_iSAID/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_resnet50_MGANet.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py  ${model}  ${Cocontrast} ${config} ${exp_dir}

echo ${arch}
echo ${config}
echo ${GPU_ID}
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --nproc_per_node=2 --master_port=23456 train.py \
        --config=${config} \
        --arch=${arch} \
        --viz \
        2>&1 | tee ${result_dir}/train-$now.log 