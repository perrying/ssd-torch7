require 'pl'
require 'image'
require 'optim'
require 'cutorch'
dofile('cfg.lua')
opt = opts.parse(arg)
cutorch.setDevice(opt.gpu)
createModel = dofile('ssd.lua')
dofile('utils.lua')
dofile('loader.lua')
dofile('train.lua')
print(opt)
print(cfg)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

local class_list = {'__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
              'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}

local class2num = {}
for i, v in pairs(class_list) do
  class2num[v] = i
end

if opt.pretrain ~= 'nil' then
  pretrain = {paths.concat(opt.pretrain, 'main_branch.t7'), paths.concat(opt.pretrain, 'branch2.t7')}
end

model = createModel(cfg.classes, cfg, pretrain)
param, gparam = model:getParameters()
conf_mat = optim.ConfusionMatrix(class_list, #class_list)
opt_conf = {
  learningRate=opt.lr,
  learningRateDecay = 0.0,
  momentum=opt.momentum,
  nesterov = true,
  dampening = 0.0,
  weightDecay=opt.wd
}

traingt, testgt, trainpath, testpath = loadPascal(opt.root, opt.cache, class2num, cfg)

train()
