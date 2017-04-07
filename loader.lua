require 'paths'

function loadPascal(root, cache_path, class2num, conf)
  local img_size = conf.imgshape
  if paths.filep(paths.concat(cache_path, 'pascal'..conf.year..'Train.t7')) and paths.filep(paths.concat(cache_path, 'paths'..conf.year..'Train.t7'))
  and paths.filep(paths.concat(cache_path, 'pascal'..conf.year..'Test.t7')) and paths.filep(paths.concat(cache_path, 'paths'..conf.year..'Test.t7')) then
    trainGT = torch.load(paths.concat(cache_path, 'pascal'..conf.year..'Train.t7'))
    testGT = torch.load(paths.concat(cache_path, 'pascal'..conf.year..'Test.t7'))
    trainPath = torch.load(paths.concat(cache_path, 'paths'..conf.year..'Train.t7'))
    testPath = torch.load(paths.concat(cache_path, 'paths'..conf.year..'Test.t7'))
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
    trainPath = {}
    testPath = {}
    for i, v in pairs(trainSet) do
      for data in io.lines(v) do
        if i == #trainSet then
          table.insert(trainPath, {'2012', data})
        else
          table.insert(trainPath, {'2007', data})
        end
      end
    end
    for data in io.lines(testSet[1]) do
      if conf.year == 2012 then
        table.insert(testPath, {'2012', data})
      else
        table.insert(testPath, {'2007', data})
      end
    end
    trainGT = {}
    testGT = {}
    for i, v in pairs(trainPath) do
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
      trainGT[v[2]] = nn.JoinTable(1):forward(tmp_data)
    end
    for i, v in pairs(testPath) do
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
      testGT[v[2]] = nn.JoinTable(1):forward(tmp_data)
    end

    torch.save(paths.concat(cache_path, 'pascal'..conf.year..'Train.t7'), trainGT)
    torch.save(paths.concat(cache_path, 'pascal'..conf.year..'Test.t7'), testGT)
    torch.save(paths.concat(cache_path, 'paths'..conf.year..'Train.t7'), trainPath)
    torch.save(paths.concat(cache_path, 'paths'..conf.year..'Test.t7'), testPath)
  end
  return trainGT, testGT, trainPath, testPath
end

-- function loadMotChallenge(root, cache_path)
--   if paths.filep(paths.concat(cache_path, 'TrainPath.t7')) and paths.filep(paths.concat(cache_path, 'TrainLabel.t7')) then
--     trainPath = torch.load(paths.concat(cache_path, 'TrainPath.t7'))
--     trainGT = torch.load(paths.concat(cache_path, 'TrainLabel.t7'))
--   end
--   local dataset_name = {}
--   for i in paths.iterfiles(paths.concat(root, 'train')) do
--     table.insert(dataset_name, i)
--   end
--   local img_name = {}
--   for i = 1, #dataset_name do
-- end
