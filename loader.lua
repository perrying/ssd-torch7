require 'paths'

function loadPascal(root, cache_path, class2num, conf)
  local img_size = conf.imgshape
  if paths.filep(paths.concat(cache_path, 'pascal'..conf.year..'Train.t7')) and paths.filep(paths.concat(cache_path, 'paths'..conf.year..'Train.t7'))
  and paths.filep(paths.concat(cache_path, 'pascal'..conf.year..'Test.t7')) and paths.filep(paths.concat(cache_path, 'paths'..conf.year..'Test.t7')) then
    train_gt = torch.load(paths.concat(cache_path, 'pascal'..conf.year..'Train.t7'))
    test_gt = torch.load(paths.concat(cache_path, 'pascal'..conf.year..'Test.t7'))
    train_path = torch.load(paths.concat(cache_path, 'paths'..conf.year..'Train.t7'))
    test_path = torch.load(paths.concat(cache_path, 'paths'..conf.year..'Test.t7'))
  else
    if conf.year == '2007' then
      trainSet = {
        paths.concat(root, 'VOC2007', 'ImageSets', 'Main', 'trainval.txt'),
        paths.concat(root, 'VOC2012', 'ImageSets', 'Main', 'trainval.txt')
      }
      testSet = {paths.concat(root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')}
    elseif conf.year == '2012' then
      trainSet = {
        paths.concat(root, 'VOC2007', 'ImageSets', 'Main', 'trainval.txt'),
        paths.concat(root, 'VOC2007', 'ImageSets', 'Main', 'test.txt'),
        paths.concat(root, 'VOC2012', 'ImageSets', 'Main', 'trainval.txt')
      }
      testSet = {paths.concat(root, 'VOC2012', 'ImageSets', 'Main', 'test.txt')}
    end
    train_path = {}
    test_path = {}
    for i, v in pairs(trainSet) do
      for data in io.lines(v) do
        if i == #trainSet then
          table.insert(train_path, {'2012', data})
        else
          table.insert(train_path, {'2007', data})
        end
      end
    end
    for data in io.lines(testSet[1]) do
      if conf.year == 2012 then
        table.insert(test_path, {'2012', data})
      else
        table.insert(test_path, {'2007', data})
      end
    end
    train_gt = {}
    test_gt = {}
    for i, v in pairs(train_path) do
      local tmp_data = {}
      local img = image.load(paths.concat(root, 'VOC'..v[1], 'JPEGImages', v[2]..'.jpg'))
      local x_len = img:size(3)
      local y_len = img:size(2)
      for v in io.lines(paths.concat(root, 'VOC'..v[1], 'Annotations', v[2]..'.txt')) do
        local label, xmin, ymin, xmax, ymax = v:match("([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)")
        xmin = xmin / x_len
        ymin = ymin / y_len
        xmax = xmax / x_len
        ymax = ymax / y_len
        table.insert(tmp_data, torch.Tensor{class2num[label], xmin, ymin, xmax, ymax}:reshape(1,5))
      end
      train_gt[v[2]] = nn.JoinTable(1):forward(tmp_data)
    end
    for i, v in pairs(test_path) do
      local tmp_data = {}
      local img = image.load(paths.concat(root, 'VOC'..v[1], 'JPEGImages', v[2]..'.jpg'))
      local x_len = img:size(3)
      local y_len = img:size(2)
      for v in io.lines(paths.concat(root, 'VOC'..v[1], 'Annotations', v[2]..'.txt')) do
        local label, xmin, ymin, xmax, ymax = v:match("([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)")
        xmin = xmin / x_len
        ymin = ymin / y_len
        xmax = xmax / x_len
        ymax = ymax / y_len
        table.insert(tmp_data, torch.Tensor{class2num[label], xmin, ymin, xmax, ymax}:reshape(1,5))
      end
      test_gt[v[2]] = nn.JoinTable(1):forward(tmp_data)
    end

    torch.save(paths.concat(cache_path, 'pascal'..conf.year..'Train.t7'), train_gt)
    torch.save(paths.concat(cache_path, 'pascal'..conf.year..'Test.t7'), test_gt)
    torch.save(paths.concat(cache_path, 'paths'..conf.year..'Train.t7'), train_path)
    torch.save(paths.concat(cache_path, 'paths'..conf.year..'Test.t7'), test_path)
  end
  return train_gt, test_gt, train_path, test_path
end
