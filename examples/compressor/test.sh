CUDA_VISIBLE_DEVICES=2 python examples/test_comp.py \
--root_dir /import/c4dm-datasets/SignalTrain_LA2A_Dataset_1.1 \
--logdir lightning_logs/version_9 \
--batch_size 128 \
--sample_rate 44100 \
--eval_subset "test" \
--eval_length 262144 \
--num_workers 8 \
--gpus 1 \
--shuffle False \
--precision 16 \
--preload True \
#--auto_lr_find 
