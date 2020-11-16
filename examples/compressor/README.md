# Analog dynamic range compressor modeling 

## Dataset

The [SignalTrain LA2A dataset](https://zenodo.org/record/3824876) (19GB) is available for download on Zenodo.
This dataset contains monophonic audio examples, with input and output targets recorded from an [LA2A dynamic range compressor](https://en.wikipedia.org/wiki/LA-2A_Leveling_Amplifier). Different parameterizations of the two controls (threshold and compress/limit) as changed for each example.
We provide a `DataLoader` in [`examples/compressor/data.py`](data.py).
<img src="https://media.uaudio.com/assetlibrary/t/e/teletronix_la2a_carousel_1_@2x_1.jpg">

In our experiments we use V1.1, which makes corrections by time aligning some files.
Download and extract this dataset before proceeding with the evaluation or retraining. 

## Pre-trained models

We provide the pre-trained model checkpoints for each of the six models for download [here](https://drive.google.com/file/d/1g1pHDVSOOtvJjIovfskX9X2295jqYD-J/view?usp=sharing) (16MB). Download this `.tgz` and extract it. 
You can run the evaluation (on the test set) with the [`examples/test_comp.py`](test_comp.py) script from the root direction after the dataset and the checkpoints have been downloaded.

We evaluate with patches of 262,144 samples (~6 seconds at 44.1 kHz) and a batch size of 128 (same as training), which requires around 12 GB of VRAM. 
We evaluate with half precision, as the models were trained in half precision as well. 
The `preload` flag will load of the audio files into RAM before training starts to run faster than continually reading them from disk. 

Below is the call we used to generate the metrics in the [paper](https://www.christiansteinmetz.com/s/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf).
```
python examples/test_comp.py \
--root_dir /path/to/SignalTrain_LA2A_Dataset_1.1 \
--logdir path/to/checkpoints/version_9 \
--batch_size 128 \
--sample_rate 44100 \
--eval_subset "test" \
--eval_length 262144 \
--num_workers 8 \
--gpus 1 \
--shuffle False \
--precision 16 \
--preload True \
```

## Retraining 
If you wish to retrain the models you can do so using the [`examples/train_comp.py`](train_comp.py) script.
Below is the call we use to train the models. 
In this case we train each of the six models for 20 epochs, which takes ~6.5h per model, for a total of ~40h when training on an NVIDIA Quadro RTX 6000. 
```
python examples/train_comp.py \
--root_dir /path/to/SignalTrain_LA2A_Dataset_1.1 \
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
```