# SSD : Single Shot Multi-Box Detector for Torch7
This is an experimental Torch7 implementation of SSD.
This code is not implemented normalization and data augmentation. (I used 1x1 convolution instead of normalization)

# Requirements
[Torch7](http://torch.ch/docs/getting-started.html), [caffe](http://caffe.berkeleyvision.org/), [cuda](https://developer.nvidia.com/cuda-downloads), [cudnn](https://developer.nvidia.com/cudnn) and

```Shell
luarocks install loadcaffe
luarocks install nninit
luarocks install optnet
```

# Usage
download PascalVOC and caffemodel, and convert data

```Shell
./DataDownloadPreprocess.sh
```

To train

```Shell
th main.lua
```

# example of output
![test1](https://github.com/perrying/ssd-torch7/blob/image/test.png)
![test3](https://github.com/perrying/ssd-torch7/blob/image/test3.png)
![test4](https://github.com/perrying/ssd-torch7/blob/image/test4.png)
![test2](https://github.com/perrying/ssd-torch7/blob/image/test2.png)


