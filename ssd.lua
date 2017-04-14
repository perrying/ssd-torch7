require 'nn';
require 'cunn';
require 'cudnn';
optnet = require 'optnet'
nninit = require 'nninit'

local function crateModel(classes, conf, pretrain)
  -- xavier initilization and bias zero
  local function xconv(ic,oc,kw,kh,sw,sh,pw,ph,type,dw,dh,relu)
    local conv
    use_relu = relu
    if type == 'N' then
      conv = cudnn.SpatialConvolution(ic, oc, kw, kh, sw, sh, pw, ph):init('weight', nninit.xavier, {dist='uniform', gain=1.1})
    elseif type == 'D' then
      local karnel = torch.randn(oc, ic, kw, kh)
      conv = nn.SpatialDilatedConvolution(ic, oc, kw, kh, sw, sh, pw, ph, pw, ph)
      nninit.xavier(nn.SpatialConvolution(ic, oc, kw, kh, sw, sh, pw, ph), karnel, {dist='uniform', gain=1.1})
      conv.weight:copy(karnel)
    end
    if cudnn.version >= 4000 then
      conv.bias = nil
      conv.gradBias = nil
    else
      conv.bias:zero()
    end
    if use_relu then
      return nn.Sequential():add(conv):add(cudnn.ReLU(true))
    else
      return conv
    end
  end

  local main
  local branch2

  -- VGG weight
  if pretrain ~= nil then
    main = torch.load(pretrain[1])
    branch2 = torch.load(pretrain[2])
  else
    main = nn.Sequential()
    branch2 = nn.Sequential()
    -- conv1 (module 1 ~ 5)
    main:add(xconv(3, 64, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(64, 64, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 300 -> 150
    -- conv2 (module 6 ~ 10)
    main:add(xconv(64, 128, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(128, 128, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 150 -> 75
    -- conv3 (module 11 ~ 17)
    main:add(xconv(128, 256, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(256, 256, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(256, 256, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 1, 1)) -- 75 -> 38
    -- conv4 (module 18 ~ 23)
    main:add(xconv(256, 512, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(512, 512, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(512, 512, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    -- conv5 (module 24 ~ 31)
    branch2:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 38 -> 19
    branch2:add(xconv(512, 512, 3, 3, 1, 1, 6, 6, 'D', 6, 6, true))
    branch2:add(xconv(512, 512, 3, 3, 1, 1, 6, 6, 'D', 6, 6, true))
    branch2:add(xconv(512, 512, 3, 3, 1, 1, 6, 6, 'D', 6, 6, true))
    branch2:add(cudnn.SpatialMaxPooling(3, 3, 1, 1, 1, 1))
    -- fc6 (module 32, 33)
    branch2:add(xconv(512, 1024, 3, 3, 1, 1, 6, 6, 'D', 6, 6))
    -- fc7 (module 34, 35)
    branch2:add(xconv(1024, 1024, 1, 1, 1, 1, 0, 0, 'N', 0, 0, true))
  end

  local branch3 = nn.Sequential()
  local branch4 = nn.Sequential()
  local branch5 = nn.Sequential()
  local branch6 = nn.Sequential()
  local subbranch1 = nn.Sequential()
  local subbranch2 = nn.Sequential()
  local subbranch3 = nn.Sequential()
  local subbranch4 = nn.Sequential()
  local subbranch5 = nn.Sequential()
  -- conv6
  branch3:add(xconv(1024, 256, 1, 1, 1, 1, 0, 0, 'N', 0, 0, true))
  branch3:add(xconv(256, 512, 3, 3, 2, 2, 1, 1, 'N', 0, 0, true)) -- 19 -> 10
  -- conv7
  branch4:add(xconv(512, 128, 1, 1, 1, 1, 0, 0, 'N', 0, 0, true))
  branch4:add(xconv(128, 256, 3, 3, 2, 2, 1, 1, 'N', 0, 0, true)) -- 10 -> 5
  -- conv8
  branch5:add(xconv(256, 128, 1, 1, 1, 1, 0, 0, 'N', 0, 0, true))
  branch5:add(xconv(128, 256, 3, 3, 1, 1, 0, 0, 'N', 0, 0, true)) -- 5 -> 3
  -- conv9
  branch6:add(xconv(256, 128, 1, 1, 1, 1, 0, 0, 'N'))
  branch6:add(xconv(128, 256, 3, 3, 1, 1, 0, 0, 'N')) -- 3 -> 1
  branch6:add(nn.ConcatTable()
  :add(nn.Sequential():add(xconv(256, 4*conf.bpc[6], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, 4)))
  :add(nn.Sequential():add(xconv(256, classes*conf.bpc[6], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, classes))))
  -- conv4 box and conf
  normalize_conv = cudnn.SpatialConvolution(512, 512, 1, 1)
  normalize_conv.weight:fill(0.001)
  if cudnn.version >= 4000 then
    normalize_conv.bias = nil
    normalize_conv.gradBias = nil
  else
    normalize_conv.bias:zero()
  end
  subbranch1:add(normalize_conv)
  subbranch1:add(nn.ConcatTable()
  :add(nn.Sequential():add(xconv(512, 4*conf.bpc[1], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, 4)))
  :add(nn.Sequential():add(xconv(512, classes*conf.bpc[1], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, classes))))
  -- fc7 box and conf
  subbranch2:add(nn.ConcatTable()
  :add(nn.Sequential():add(xconv(1024, 4*conf.bpc[2], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, 4)))
  :add(nn.Sequential():add(xconv(1024, classes*conf.bpc[2], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, classes))))
  -- conv6 box and conf
  subbranch3:add(nn.ConcatTable()
  :add(nn.Sequential():add(xconv(512, 4*conf.bpc[3], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, 4)))
  :add(nn.Sequential():add(xconv(512, classes*conf.bpc[3], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, classes))))
  -- conv7 box and conf
  subbranch4:add(nn.ConcatTable()
  :add(nn.Sequential():add(xconv(256, 4*conf.bpc[4], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, 4)))
  :add(nn.Sequential():add(xconv(256, classes*conf.bpc[4], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, classes))))
  -- conv8 box and conf
  subbranch5:add(nn.ConcatTable()
  :add(nn.Sequential():add(xconv(256, 4*conf.bpc[5], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, 4)))
  :add(nn.Sequential():add(xconv(256, classes*conf.bpc[5], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
  :add(nn.Transpose({2,3},{3,4}))
  :add(nn.Reshape(-1, classes))))
  -- join all branches
  main:add(nn.ConcatTable():add(subbranch1)
  :add(branch2:add(nn.ConcatTable():add(subbranch2)
  :add(branch3:add(nn.ConcatTable():add(subbranch3)
  :add(branch4:add(nn.ConcatTable():add(subbranch4)
  :add(branch5:add(nn.ConcatTable():add(subbranch5)
  :add(branch6)))))))))):add(nn.FlattenTable())
  -- transform
  main:add(nn.ConcatTable():add(nn.SelectTable(1))
  :add(nn.SelectTable(3))
  :add(nn.SelectTable(5))
  :add(nn.SelectTable(7))
  :add(nn.SelectTable(9))
  :add(nn.SelectTable(11))
  :add(nn.SelectTable(2))
  :add(nn.SelectTable(4))
  :add(nn.SelectTable(6))
  :add(nn.SelectTable(8))
  :add(nn.SelectTable(10))
  :add(nn.SelectTable(12)))

  main:add(nn.ConcatTable()
  :add(nn.Sequential()
  :add(nn.NarrowTable(1, 6))
  :add(nn.JoinTable(2)))
  :add(nn.Sequential()
  :add(nn.NarrowTable(7, 12))
  :add(nn.JoinTable(2))))

  main = main:cuda()
  local inp = torch.randn(1, 3, conf.imgshape, conf.imgshape):cuda()
  local opts = {inplace=true, mode='training'}
  optnet.optimizeMemory(main, inp, opts)
  return main
end
return crateModel
