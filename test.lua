function test()
  model:evaluate()
  for i = 1, #testpath, opt.batchsize do
    local inputs = torch.Tensor(math.min(opt.batchsize, #testpath-i), 3, cfg.imgshape, cfg.imgshape)
    local gt_labels = {}
    local gt_bboxes = {}
    for j = 1, math.min(opt.batchsize, #testpath-i) do
      local img = image.load(paths.concat(opt.root, 'VOC'..trainpath[i][1], 'JPEGImages', trainpath[i][2]..'.jpg'))
      img = image.scale(img, cfg.imgshape, cfg.imgshape):reshape(1, 3, cfg.imgshape, cfg.imgshape)
      -- channel RGB to BGR
      local tmp_r = img[{{},{1},{},{}}]
      local tmp_b = img[{{},{3},{},{}}]
      img[{{},{1},{},{}}] = tmp_b
      img[{{},{3},{},{}}] = tmp_r
      inputs[j] = img
      local info = testgt[testpath[i][2]]
      local label = info:narrow(2,1,1)
      local box = info:narrow(2,2,4):reshape(info:size(1),4)
      table.insert(gt_bboxes, box)
      table.insert(gt_labels, label)
    end
    local outputs = model:forward(inputs:cuda())
    local loc_preds = outputs[1]
    local conf_preds = outputs[2]
    local bboxes, classes, score = Detect(loc_preds, conf_preds, cfg.nms_threshold, cfg.iou_threshold)
    local pred, gt = Evaluate(bboxes, classes, gt_bboxes, gt_labels)
  end
  print(conf_mat)
end
