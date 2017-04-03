require 'nn'
require 'cudnn'
require 'cunn'
require 'loadcaffe'

local net = loadcaffe.load('../deploy.prototxt', 'VGG_ILSVRC_16_layers_fc_reduced.caffemodel', 'cudnn')

local branch = nn.Sequential()

for i = 1, #net.modules - 1 do -- except output layer
  if i == 17 then
    branch:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
  elseif i == 25 or i == 27 or i == 29 then
    local Dconv = nn.SpatialDilatedConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1, 1)
    Dconv.weight:copy(net:get(i).weight)
    Dconv.bias:copy(net:get(i).bias)
    branch:add(Dconv)
  elseif i == 31 then
    branch:add(cudnn.SpatialMaxPooling(3, 3, 1, 1, 1, 1))
  elseif i == 32 then
    local conv = nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1)
    conv.weight:copy(net:get(i).weight)
    conv.bias:copy(net:get(i).bias)
    branch:add(conv)
  elseif i ~= 34 and i ~= 37 then -- except Dropout layer
    branch:add(net:get(i))
  end

  if i == 23 then
    torch.save('main_branch.t7', branch)
    branch = nn.Sequential()
  end
end

torch.save('branch2.t7', branch)
