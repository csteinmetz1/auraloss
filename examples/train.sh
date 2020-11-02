python examples/train_denoise.py \
--root_dir ./data/MiniLibriMix \
--sample_rate 8000 \
--train_length 16384 \
--eval_length 32768 \
--num_workers 0 \
--depthwise True \
--channel_width 16 \
--dilation_growth 2 \
--auto_lr_find 
