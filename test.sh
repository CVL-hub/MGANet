PARTITION=Segmentation



GPU_ID=1

dataset=iSAID # iSAID/ LoveDA

exp_name=split0

arch=MGANet
visualize=False

net=resnet50 # vgg resnet50

exp_dir=exp/MGANet_iSAID/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_resnet50_MGANet.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp test.sh test.py ${config} ${exp_dir}

echo ${arch}
echo ${config}
echo ${visualize}

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u test.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/test-$now.log