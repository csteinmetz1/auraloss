# auraloss
A collection of audio focused loss functions in PyTorch.

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

# Loss functions

We categorize all loss functions as either time-domain or frequency domain approaches. 

## Time-domain

- Error-to-signal ratio (ESR)
- DC error 
- Logcosh
- SI-SDR

## Frequency-domain

- Spectral Convergence 
- Log STFT Magnitude
- Aggregate STFT 
- Multi-Resolution STFT 
- Sum and difference (stereo)

## References
https://zenodo.org/record/4091379

Would be nice to have math (LaTeX) in the docstrings. 

# Cite
If you use this code in your work please consider citing us.
```
```
