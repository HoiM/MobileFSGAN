# Migrating Face Swap to Mobile Devices: A lightweight Framework and A Supervised Training Solution

### Introduction

This is the repository for the source code of the papar: Migrating Face Swap to Mobile Devices: A lightweight Framework and A Supervised Training Solution

Accepted to [ICME 2022](http://2022.ieeeicme.org/)

[arXiv]()

### Preferable configurations

torch==1.6.0

torchvision==0.7.0

cuda version: 10.2

### How to use the code

First of all, I strongly recommend implementing your own `dataset` class for loading training data. Since my implemenation is based the structure of my `data` folder. 
```
class FaceShifterDataset(torch.utils.data.TensorDataset):
    def __init__(self):
        super(FaceShifterDataset, self).__init__()
        # TODO: add your code

    def __getitem__(self, item):
        # TODO: add your code
        return Xs, Xt, GT, with_gt, src_as_true

    def __len__(self):
        # TODO: add your code
        return len(self.data_list)
```
Basically, the `__getitem__` function returns five elements:
* `Xs`: the source image (shape: (3, 256, 256), values in (-1, 1))
* `Xt`: the target image
* `GT`: the ground truth image. If none, an equivalent tensor with all values being -1. 
* `with_gt`: whether the training data has a ground truth label (data type: torch.float32).
* `src_as_true`: used to indicate whether the source or the target is a real image. Selected real images are used to feed the discrimiator.

After that, reset the arugments inside `main.py` and refer to `run_training.sh`. Note that my implementation uses `DistributedDataParallel` on 4 GPUs within one machine. Please adjust the settings based on your hardware. 

### Performance

##### Qualitative performance

![Selected results](./params/Performance.png)

