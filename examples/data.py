import os
import sys
import glob
import torch 
import torchaudio
import numpy as np
import soundfile as sf
torchaudio.set_audio_backend("sox_io")

class LibriMixDataset(torch.utils.data.Dataset):
    """ LibriMix dataset."""
    def __init__(self, root_dir, subset="train", length=16384, noisy=False):
        """
        Args:
            root_dir (str): Path to the preprocessed LibriMix files.
            subset (str, optional): Pull data either from "train", "val", or "test" subsets. (Default: "train")
            length (int, optional): Number of samples in the returned examples. (Default: 40)
            noise (bool, optional): Use mixtures with additive noise, otherwise anechoic mixes. (Default: False)
        """
        self.root_dir = root_dir
        self.subset = subset
        self.length = length
        self.noisy = noisy

        # set the mix directory if we want clean or noisy mixes as input
        self.mix_dir = "mix_both" if self.noisy else "mix_clean"

        # get all the files in the mix directory first
        self.files = glob.glob(os.path.join(self.root_dir, self.subset, self.mix_dir, "*.wav"))
        self.hours = 0 # total number of hours of data in the subset

        # loop over files to count total length
        for filename in self.files:
            si, ei = torchaudio.info(filename)
            self.hours += (si.length / si.rate) / 3600 

        # we then want to remove the path and extract just file ids
        self.files = [os.path.basename(filename) for filename in self.files]
        print(f"Located {len(self.files)} examples totaling {self.hours:0.1f} hr in the {self.subset} subset.")

    def __len__(self):
        return 32 #len(self.files)

    def __getitem__(self, idx):

        eid = self.files[idx]

        # use torchaudio to load them, which should be pretty fast
        s1,    sr = torchaudio.load(os.path.join(self.root_dir, self.subset, "s1", eid))
        s2,    sr = torchaudio.load(os.path.join(self.root_dir, self.subset, "s2", eid))
        noise, sr = torchaudio.load(os.path.join(self.root_dir, self.subset, "noise", eid))
        mix,   sr = torchaudio.load(os.path.join(self.root_dir, self.subset, self.mix_dir, eid))

        # get the length of the current file in samples
        si, ei = torchaudio.info(os.path.join(self.root_dir, self.subset, "s1", eid))

        # pad if too short
        if si.length < self.length:
            pad_length = self.length - si.length
            s1 = torch.nn.functional.pad(s1, (0,pad_length))
            s2 = torch.nn.functional.pad(s2, (0,pad_length))
            noise = torch.nn.functional.pad(noise, (0,pad_length))
            mix = torch.nn.functional.pad(mix, (0,pad_length))
            si.length = self.length

        # choose a random patch of `length` samples for training
        start_idx = np.random.randint(0, si.length - self.length + 1)
        stop_idx = start_idx + self.length

        # extract these patches from each sample 
        s1    = s1[0,start_idx:stop_idx].unsqueeze(dim=0)
        s2    = s2[0,start_idx:stop_idx].unsqueeze(dim=0)
        noise = noise[0,start_idx:stop_idx].unsqueeze(dim=0)
        mix   = mix[0,start_idx:stop_idx].unsqueeze(dim=0)

        return s1, s2, noise, mix

class SignalTrainLA2ADataset(torch.utils.data.Dataset):
    """ SignalTrain LA2A dataset. Source: [10.5281/zenodo.3824876](https://zenodo.org/record/3824876)."""
    def __init__(self, root_dir, subset="train", length=16384, preload=False, half=True, use_soundfile=False):
        """
        Args:
            root_dir (str): Path to the root directory of the SignalTrain dataset.
            subset (str, optional): Pull data either from "train", "val", or "test" subsets. (Default: "train")
            length (int, optional): Number of samples in the returned examples. (Default: 40)
            preload (bool, optional): Read in all data into RAM during init. (Default: False)
            half (bool, optional): Store the float32 audio as float16. (Default: True)
            use_soundfile (bool, optional): Use the soundfile library to load instead of torchaudio. (Default: False)
        """
        self.root_dir = root_dir
        self.subset = subset
        self.length = length
        self.preload = preload
        self.half = half
        self.use_soundfile = use_soundfile

        # get all the target files files in the directory first
        self.target_files = glob.glob(os.path.join(self.root_dir, self.subset.capitalize(), "target_*.wav"))
        self.input_files  = glob.glob(os.path.join(self.root_dir, self.subset.capitalize(), "input_*.wav"))

        self.examples = [] 
        self.hours = 0  # total number of hours of data in the subset

        # ensure that the sets are ordered correctlty
        self.target_files.sort()
        self.input_files.sort()

        # get the parameters 
        self.params = [(float(f.split("__")[1].replace(".wav","")), float(f.split("__")[2].replace(".wav",""))) for f in self.target_files]

        # loop over files to count total length
        for idx, (tfile, ifile, params) in enumerate(zip(self.target_files, self.input_files, self.params)):

            ifile_id = int(os.path.basename(ifile).split("_")[1])
            tfile_id = int(os.path.basename(tfile).split("_")[1])
            if ifile_id != tfile_id:
                raise RuntimeError(f"Found non-matching file ids: {ifile_id} != {tfile_id}! Check dataset.")

            md = torchaudio.info(tfile)
            self.hours += (md.num_frames / md.sample_rate) / 3600 
            num_frames = md.num_frames

            if self.preload:
                sys.stdout.write(f"* Pre-loading... {idx+1:3d}/{len(self.target_files):3d} ...\r")
                sys.stdout.flush()
                input, sr  = self.load(ifile)
                target, sr = self.load(tfile)

                num_frames = int(np.min([input.shape[-1], target.shape[-1]]))
                if input.shape[-1] != target.shape[-1]:
                    print(os.path.basename(ifile), input.shape[-1], os.path.basename(tfile), target.shape[-1])
                    raise RuntimeError("Found potentially corrupt file!")
                if self.half:
                    input = input.half()
                    target = target.half()
            else:
                input = None
                target = None

            # create one entry for each patch
            for n in range((num_frames // self.length) - 1):
                offset = int(n * self.length)
                end = offset + self.length
                self.examples.append({"idx": idx, 
                                      "target_file" : tfile, 
                                      "input_file" : ifile, 
                                      "input_audio" : input[:,offset:end],
                                      "target_audio" : target[:,offset:end],
                                      "params" : params, 
                                      "offset": offset, 
                                      "frames" : num_frames})

        # we then want to get the input files
        print(f"Located {len(self.examples)} examples totaling {self.hours:0.1f} hr in the {self.subset} subset.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.preload:
            audio_idx = self.examples[idx]["idx"]
            offset = self.examples[idx]["offset"]
            input = self.examples[idx]["input_audio"]
            target = self.examples[idx]["target_audio"]
        else:
            offset = self.examples[idx]["offset"] 
            input, sr  = torchaudio.load(self.examples[idx]["input_file"], 
                                        num_frames=self.length, 
                                        frame_offset=offset, 
                                        normalize=False)
            target, sr = torchaudio.load(self.examples[idx]["target_file"], 
                                        num_frames=self.length, 
                                        frame_offset=offset, 
                                        normalize=False)
            if self.half:
                input = input.half()
                target = target.half()

        # at random with p=0.5 flip the phase 
        if np.random.rand() > 0.5:
            input *= -1
            target *= -1

        # then get the tuple of parameters
        params = torch.tensor(self.examples[idx]["params"]).unsqueeze(0)
        params[:,1] /= 100

        return input, target, params

    def load(self, filename):
        if self.use_soundfile:
            x, sr = sf.read(filename, always_2d=True)
            x = torch.tensor(x.T)
        else:
            x, sr = torchaudio.load(filename, normalize=False)
        return x, sr