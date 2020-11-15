CUDA_VISIBLE_DEVICES=2 python examples/train_comp.py \
--root_dir /import/c4dm-datasets/SignalTrain_LA2A_Dataset_1.1 \
--max_epochs 20 \
--batch_size 128 \
--sample_rate 44100 \
--train_length 32768 \
--eval_length 262144 \
--num_workers 8 \
--kernel_size 15 \
--channel_width 32 \
--dilation_growth 2 \
--lr 0.001 \
--gpus 1 \
--shuffle True \
--precision 16 \
--preload True \
#--auto_lr_find 
