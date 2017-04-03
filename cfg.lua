lapp = require 'pl.lapp'
-- network config
cfg = {
  nmap = 6, -- number of feature maps
  msize = {38, 19, 10, 5, 3, 1}, -- feature matp size
  bpc = {4, 6, 6, 6, 4, 4}, -- number of default boxes per cell
  s_min = 20, -- minimum box size
  s_max = 90, -- maximum box size
  aratio = {1, '1', 2, 1/2, 3, 1/3},  -- aspect retio
  variance = {0.1, 0.1, 0.2, 0.2},
  steps = {8, 16, 32, 64, 100, 300},
  imgshape = 300, -- input image size
  iou_threshold = 0.5, -- iou threshold
  classes = 21, -- classes
  year = '2007' -- 2007 or 2012
}

opts = {}
-- parameters and others config
function opts.parse(arg)
  opt = lapp [[
    Command line options:
    Training Related:
    --lr         (default 1e-3)                    learning rate
    --momentum   (default 0.9)                     momentum
    --wd         (default 0.0005)                  weight decay
    --snap       (default 10000)                   snapshot
    --iter       (default 120000)                   iterations
    --batchsize  (default 32)                      mini-batch size
    --test       (default 10000)                   test span
    --disp       (default 100)                     display span
    --output     (default ./output)                output directory
    --root       (default data/VOCdevkit)          dataset root directory
    --cache      (default ./cache)                 cache file directory
    --gpu        (default 1)                       gpu id
    --pretrain   (default ./caffemodel)                     pretrain model root directory
    --seed       (default 1)
  ]]
  return opt
end
