mkdir output
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch \
        --master_port 42222 \
        --nproc_per_node=4 main.py \
        > output/nohup.out &
