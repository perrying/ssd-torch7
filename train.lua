function train()
  model:training()
  local index = 1
  local mean_loss = 0
  local mean_loc_loss = 0
  local mean_conf_loss = 0
  local imgs = torch.FloatTensor(opt.batchsize, 3, cfg.imgshape, cfg.imgshape)
  local gt_bboxes = {}
  local gt_labels = {}
  local shuffle = torch.randperm(#trainpath)
  -- training
  for i = 1, opt.iter * opt.batchsize do
    local img = image.load(paths.concat(opt.root, 'VOC'..trainpath[shuffle[index]][1], 'JPEGImages', trainpath[shuffle[index]][2]..'.jpg'))
    img = image.scale(img, cfg.imgshape, cfg.imgshape):reshape(1, 3, cfg.imgshape, cfg.imgshape)
    -- channel RGB to BGR
    local tmp_r = img[{{},{1},{},{}}]
    local tmp_b = img[{{},{3},{},{}}]
    img[{{},{1},{},{}}] = tmp_b
    img[{{},{3},{},{}}] = tmp_r
    -- get gt
    local info = traingt[trainpath[shuffle[index]][2]]
    local label = info:narrow(2,1,1)
    local box = info:narrow(2,2,4):reshape(info:size(1),4)
    imgs[(index-1) % opt.batchsize + 1] = img
    table.insert(gt_bboxes, box)
    table.insert(gt_labels, label)
    -- batch forward and backward
    if index % opt.batchsize == 0 then
      gparam:zero()
      local outputs = model:forward(imgs:cuda())
      local loc_preds = outputs[1]
      local conf_preds = outputs[2]
      local loss = 0
      local sum_loc_loss = 0
      local sum_conf_loss = 0
      local loc_grads = torch.Tensor(loc_preds:size())
      local conf_grads = torch.Tensor(conf_preds:size())
      -- calc gradient
      for j = 1, opt.batchsize do
        local loc_grad, conf_grad, loc_loss, conf_loss = MultiBoxLoss(loc_preds[j], conf_preds[j], gt_bboxes[j], gt_labels[j], cfg)
        loss = loss + (loc_loss + conf_loss)
        sum_loc_loss = sum_loc_loss + loc_loss
        sum_conf_loss = sum_conf_loss + conf_loss
        loc_grads[j] = loc_grad
        conf_grads[j] = conf_grad
      end
      loss = loss / opt.batchsize
      sum_loc_loss = sum_loc_loss / opt.batchsize
      sum_conf_loss = sum_conf_loss / opt.batchsize
      mean_loss = mean_loss + loss
      mean_loc_loss = mean_loc_loss + sum_loc_loss
      mean_conf_loss = mean_conf_loss + sum_conf_loss
      -- backward
      model:backward(imgs:cuda(), {loc_grads, conf_grads})
      gparam:div(opt.batchsize)
      local function feval() return loss, gparam end
      -- parameter update
      optim.sgd(feval, param, opt_conf)
      gt_bboxes = {}
      gt_labels = {}
      collectgarbage()
    end
    -- save model
    if i % (opt.snap * opt.batchsize) == 0 then
      torch.save(paths.concat(opt.output, 'model'..(i/opt.batchsize)..'iter.t7'), model)
    end
    -- learning rate decay
    if cfg.year == '2007' and (i / opt.batchsize == 80000 or i / opt.batchsize == 100000) then
      opt.lr = opt.lr * 0.1
      opt_conf.learningRate = opt.lr
    elseif cfg.year == '2012' and i / opt.batchsize == 60000 then
      opt.lr = opt.lr * 0.1
      opt_conf.learningRate = opt.lr
    end
    -- index reset
    if index == #trainpath then
      print('1 epoch finish')
      index = 0
      shuffle = torch.randperm(#trainpath)
    end
    -- next index
    index = index + 1
    -- display
    if i % (opt.disp * opt.batchsize) == 0 then
      print('iter : '..i/opt.batchsize..'   mean error : '..mean_loss/opt.disp)
      print('mean l1 loss : '..mean_loc_loss/opt.disp)
      print('mean cross entropy : '..mean_conf_loss/opt.disp)
      mean_loss = 0
      mean_loc_loss = 0
      mean_conf_loss = 0
      print(conf_mat)
      local dataNum = conf_mat.mat:sum(2)
      local tp = dataNum:narrow(1,2,cfg.classes - 1):sum()
      local fp = dataNum[1]:sum()
      print('TP : '..tp..'  FP : '..fp)
      conf_mat:zero()
    end
  end
end
