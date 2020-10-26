# auraloss
Collection of audio focused loss functions in PyTorch.

## Setup

```
pip install git+https://github.com/csteinmetz1/auraloss
```

## Usage

```python
import torch
import auraloss

sisdr = auraloss.SISDRLoss()

x = torch.rand(8,1,44100)
y = torch.rand(8,1,44100)

loss = sisdr(x, y)

```

## Details

All loss functions support multi-channel audio. 

## Cite
```
```
