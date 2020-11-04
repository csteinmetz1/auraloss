python examples/train_comp.py \
--root_dir /Volumes/SteinmetzAlpha/Datasets/msc/SignalTrain_LA2A_Dataset \
--batch_size 2 \
--sample_rate 44100 \
--train_length 16384 \
--eval_length 262144 \
--num_workers 0 \
--kernel_size 15 \
--channel_width 32 \
--dilation_growth 2 \
--lr 0.004 \
--train_loss mrstft \
--shuffle True \
#--auto_lr_find 
