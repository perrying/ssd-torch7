require 'nn';
require 'cunn';

loc_loss_func = nn.SmoothL1Criterion():cuda()
loc_loss_func.sizeAverage = false
conf_loss_func = nn.CrossEntropyCriterion():cuda()
conf_loss_func.nll.sizeAverage = false

function EncodeBBox(bbox, prior_bboxes, variance)
  local prior_center_x = torch.add(prior_bboxes[{{}, {1}}], prior_bboxes[{{}, {3}}]):div(2)
  local prior_center_y = torch.add(prior_bboxes[{{}, {2}}], prior_bboxes[{{}, {4}}]):div(2)
  local prior_width = torch.csub(prior_bboxes[{{}, {3}}], prior_bboxes[{{}, {1}}])
  local prior_height = torch.csub(prior_bboxes[{{}, {4}}], prior_bboxes[{{}, {2}}])
  local bbox_center_x = torch.add(bbox[{{}, {1}}], bbox[{{}, {3}}]):div(2)
  local bbox_center_y = torch.add(bbox[{{}, {2}}], bbox[{{}, {4}}]):div(2)
  local bbox_width = torch.csub(bbox[{{}, {3}}], bbox[{{}, {1}}])
  local bbox_height = torch.csub(bbox[{{}, {4}}], bbox[{{}, {2}}])
  local encode_bbox_xmin, encode_bbox_ymin, encode_bbox_xmax, encode_bbox_ymax
  if variance == nil then
    encode_bbox_xmin = torch.cdiv(torch.csub(bbox_center_x, prior_center_x), prior_width):view(-1, 1)
    encode_bbox_ymin = torch.cdiv(torch.csub(bbox_center_y, prior_center_y), prior_height):view(-1, 1)
    encode_bbox_xmax = torch.log(torch.cdiv(bbox_width, prior_width)):view(-1, 1)
    encode_bbox_ymax = torch.log(torch.cdiv(bbox_height, prior_height)):view(-1, 1)
  else
    encode_bbox_xmin = (torch.cdiv(torch.csub(bbox_center_x, prior_center_x), prior_width) / variance[1]):view(-1, 1)
    encode_bbox_ymin = (torch.cdiv(torch.csub(bbox_center_y, prior_center_y), prior_height) / variance[2]):view(-1, 1)
    encode_bbox_xmax = (torch.log(torch.cdiv(bbox_width, prior_width)) / variance[3]):view(-1, 1)
    encode_bbox_ymax = (torch.log(torch.cdiv(bbox_height, prior_height)) / variance[4]):view(-1, 1)
  end
  return torch.cat(torch.cat(encode_bbox_xmin, encode_bbox_ymin, 2), torch.cat(encode_bbox_xmax, encode_bbox_ymax, 2), 2)
end

function DecodeBBox(bbox, prior_bboxes, variance)
  local prior_center_x = torch.add(prior_bboxes[{{}, {1}}], prior_bboxes[{{}, {3}}]):div(2)
  local prior_center_y = torch.add(prior_bboxes[{{}, {2}}], prior_bboxes[{{}, {4}}]):div(2)
  local prior_width = torch.csub(prior_bboxes[{{}, {3}}], prior_bboxes[{{}, {1}}])
  local prior_height = torch.csub(prior_bboxes[{{}, {4}}], prior_bboxes[{{}, {2}}])
  local decode_bbox_center_x, decode_bbox_center_y, decode_bbox_width, decode_bbox_height
  if variance == nil then
    decode_bbox_center_x = torch.add(torch.cmul(bbox[{{}, {1}}], prior_width), prior_center_x)
    decode_bbox_center_y = torch.add(torch.cmul(bbox[{{}, {2}}], prior_height), prior_center_y)
    decode_bbox_width = torch.cmul(torch.exp(bbox[{{}, {3}}]), prior_width)
    decode_bbox_height = torch.cmul(torch.exp(bbox[{{}, {4}}]), prior_width)
  else
    decode_bbox_center_x = torch.add(torch.cmul(bbox[{{}, {1}}], prior_width) * variance[1], prior_center_x)
    decode_bbox_center_y = torch.add(torch.cmul(bbox[{{}, {2}}], prior_height) * variance[2], prior_center_y)
    decode_bbox_width = torch.cmul(torch.exp(bbox[{{}, {3}}] * variance[3]), prior_width)
    decode_bbox_height = torch.cmul(torch.exp(bbox[{{}, {4}}] * variance[4]), prior_width)
  end
  local xmin = torch.csub(decode_bbox_center_x, decode_bbox_width/2):view(-1, 1)
  local ymin = torch.csub(decode_bbox_center_y, decode_bbox_height/2):view(-1, 1)
  local xmax = torch.add(decode_bbox_center_x, decode_bbox_width/2):view(-1, 1)
  local ymax = torch.add(decode_bbox_center_y, decode_bbox_height/2):view(-1, 1)
  return torch.cat(torch.cat(xmin, ymin, 2), torch.cat(xmax, ymax, 2), 2)
end

function GetPriorBBoxes(cfg)
  local function _calc_width(ar, min_size, max_size)
    if ar == '1' then
      return math.sqrt(min_size * max_size)
    else
      return min_size * math.sqrt(ar)
    end
  end
  local function _calc_height(ar, min_size, max_size)
    if ar == '1' then
      return math.sqrt(min_size * max_size)
    else
      return min_size / math.sqrt(ar)
    end
  end
  local min_sizes, max_sizes = CalcPriorBBoxParam(cfg)
  local map_num = cfg.nmap or 6
  local map_size = cfg.msize or {38, 19, 10, 5, 3, 1}
  local img_size = cfg.imgshape or 300
  local box_per_cell = cfg.bpc or {4, 6, 6, 6, 4, 4}
  local ar = cfg.aratio or {1, '1', 2, 1/2, 3, 1/3}
  local steps = cfg.steps or {8, 16, 32, 64, 100, 300}
  local prior_bboxes = {}
  for k = 1, map_num do
    local step_w = steps[k]
    local step_h = steps[k]
    local tmp_prior_bboxes = torch.zeros(map_size[k], map_size[k], box_per_cell[k]*4)
    for h = 1, map_size[k] do
      for w = 1, map_size[k] do
        local center_x = ((w-1) + 0.5) * step_w
        local center_y = ((h-1) + 0.5) * step_h
        for b = 1, box_per_cell[k] do
          local box_width = _calc_width(ar[b], min_sizes[k], max_sizes[k])
          local box_height = _calc_height(ar[b], min_sizes[k], max_sizes[k])
          tmp_prior_bboxes[h][w][(b-1)*4+1] = math.min(math.max((center_x - box_width / 2) / img_size, 0), 1) -- xmin
          tmp_prior_bboxes[h][w][(b-1)*4+2] = math.min(math.max((center_y - box_height / 2) / img_size, 0), 1) -- ymin
          tmp_prior_bboxes[h][w][(b-1)*4+3] = math.min(math.max((center_x + box_width / 2) / img_size, 0), 1) -- xmax
          tmp_prior_bboxes[h][w][(b-1)*4+4] = math.min(math.max((center_y + box_height / 2) / img_size, 0), 1) -- ymax
        end
      end
    end
    table.insert(prior_bboxes, tmp_prior_bboxes:view(-1, 4))
  end
  return nn.JoinTable(1):forward(prior_bboxes)
end

function CalcPriorBBoxParam(cfg)
  local min_dim = cfg.imgshape or 300
  local min_ratio = cfg.s_min or 20
  local max_ratio = cfg.s_max or 90
  local layer_num = cfg.nmap or 6
  local step = math.floor((max_ratio - min_ratio) / (layer_num - 2))
  local min_sizes = {min_dim * 10 / 100.} -- first layer's scale
  local max_sizes = {min_dim * 20 / 100.}
  for ratio = min_ratio, max_ratio, step do
    table.insert(min_sizes, min_dim * ratio / 100)
    table.insert(max_sizes, min_dim * (ratio + step) / 100)
  end
  return min_sizes, max_sizes
end

function EncodeLocPrediction(loc_preds, prior_bboxes, gt_locs, match_indices, cfg)
  local loc_gt_data = EncodeBBox(gt_locs:index(1, match_indices:nonzero():view(-1)), prior_bboxes:index(1, match_indices:nonzero():view(-1)), cfg.variance)
  local loc_pred_data = loc_preds:index(1, match_indices:nonzero():view(-1))
  return loc_gt_data, loc_pred_data
end

function EncodeConfPrediction(conf_preds, match_indices, neg_indices)
  local num_matches = match_indices:nonzero():size(1)
  local num_samples = num_matches + neg_indices:size(1)
  local conf_gt_data = torch.zeros(num_samples)
  conf_gt_data[{{1,num_matches}}] = match_indices[match_indices:ne(0)]
  conf_gt_data[{{num_matches+1,-1}}] = 1
  local match_preds = conf_preds:index(1, match_indices:nonzero():view(-1))
  local neg_preds = conf_preds:index(1, match_indices:eq(0):nonzero():view(-1)):index(1, neg_indices)
  local conf_pred_data = torch.cat(match_preds, neg_preds, 1)
  return conf_gt_data, conf_pred_data
end

function BBoxSize(bbox)
  local sizes = torch.zeros(bbox:size(1))
  local idx = torch.cmul(bbox[{{}, {3}}]:gt(bbox[{{}, {1}}]), bbox[{{}, {4}}]:gt(bbox[{{}, {2}}])):view(-1)
  local width = torch.csub(bbox[{{}, {3}}][idx], bbox[{{}, {1}}][idx])
  local height = torch.csub(bbox[{{}, {4}}][idx], bbox[{{}, {2}}][idx])
  sizes[idx] = torch.cmul(width, height)
  return sizes:float()
end

function JaccardOverlap(bbox, gtbbox)
  local xmin = torch.cmax(bbox[{{}, {1}}], gtbbox[1])
  local ymin = torch.cmax(bbox[{{}, {2}}], gtbbox[2])
  local xmax = torch.cmin(bbox[{{}, {3}}], gtbbox[3])
  local ymax = torch.cmin(bbox[{{}, {4}}], gtbbox[4])
  local width = torch.cmax(torch.csub(xmax, xmin),0)
  local height = torch.cmax(torch.csub(ymax, ymin),0)
  local intersect_size = torch.cmul(width, height)
  local bbox_size = BBoxSize(bbox)
  local gtbbox_size = BBoxSize(gtbbox:reshape(1,4))
  return torch.cdiv(intersect_size, torch.csub(torch.add(bbox_size, gtbbox_size[1]), intersect_size))
end

function MatchingBBoxes(prior_bboxes, gt_bboxes, gt_labels, cfg)
  local match_indices = torch.zeros(prior_bboxes:size(1))
  local match_overlaps = torch.zeros(prior_bboxes:size(1))
  local gt_locs = torch.zeros(prior_bboxes:size())
  for i = 1, gt_bboxes:size(1) do
    local overlaps = JaccardOverlap(prior_bboxes, gt_bboxes[i]):view(-1)
    local max_overlap, max_idx = overlaps:max(1)
    if max_overlap[1] < cfg.iou_threshold then
      match_overlaps[max_idx[1]] = 1.1
      match_indices[max_idx[1]] = gt_labels[i][1]
      gt_locs[max_idx[1]] = gt_bboxes[i]
    else
      local idx = torch.cmul(overlaps:ge(cfg.iou_threshold), overlaps:gt(match_overlaps))
      match_overlaps[idx] = overlaps[idx]
      match_indices[idx] = gt_labels[i][1]
      gt_locs[{{}, {1}}][idx] = gt_bboxes[i][1]
      gt_locs[{{}, {2}}][idx] = gt_bboxes[i][2]
      gt_locs[{{}, {3}}][idx] = gt_bboxes[i][3]
      gt_locs[{{}, {4}}][idx] = gt_bboxes[i][4]
    end
  end
  return match_indices, gt_locs
end

function MineHardExamples(conf_preds, match_indices)
  local num_matches = match_indices:nonzero():size(1)
  local num_sel = math.min(num_matches * 3, match_indices:eq(0):sum())
  -- calc loss
  local neg_loss = -torch.log(torch.cdiv(
  torch.exp(conf_preds[{{},{1}}][match_indices:eq(0)]),
  torch.exp(conf_preds):sum(2)[match_indices:eq(0)]))
  -- get topk
  local topk, neg_indices = neg_loss:topk(num_sel, true)
  return neg_indices
end

function GetGradient(loc_gt_data, loc_pred_data, conf_gt_data, conf_pred_data, match_indices, neg_indices, prior_bboxes_shape, cfg)
  local num_matches = match_indices:nonzero():size(1)
  local match_indices_tensor = match_indices:nonzero():view(-1)
  local not_match_indices_tensor = match_indices:eq(0):nonzero():view(-1)
  local loc_loss = loc_loss_func:forward(loc_pred_data:cuda(), loc_gt_data:cuda()) / num_matches
  local loc_grad = (loc_loss_func:backward(loc_pred_data:cuda(), loc_gt_data:cuda()) / num_matches):float()
  local conf_loss = conf_loss_func:forward(conf_pred_data:cuda(), conf_gt_data:cuda()) / num_matches
  local conf_grad = (conf_loss_func:backward(conf_pred_data:cuda(), conf_gt_data:cuda()) / num_matches):float()
  if conf_mat ~= nil then
    conf_mat:batchAdd(conf_pred_data, conf_gt_data)
  end
  local loc_dE_do = torch.zeros(prior_bboxes_shape)
  local conf_dE_do = torch.zeros(prior_bboxes_shape[1], cfg.classes)
  for i = 1, num_matches do
    loc_dE_do[match_indices_tensor[i]] = loc_grad[i]:float()
    conf_dE_do[match_indices_tensor[i]] = conf_grad[i]:float()
  end
  for i = num_matches + 1, num_matches + neg_indices:size(1) do
    local neg_index = not_match_indices_tensor[neg_indices[i-num_matches]]
    conf_dE_do[neg_index] = conf_grad[i]
  end
  return loc_dE_do, conf_dE_do, loc_loss, conf_loss
end

function MultiBoxLoss(loc_preds, conf_preds, gt_bboxes, gt_labels, cfg)
  local prior_bboxes = GetPriorBBoxes(cfg)
  local match_indices, gt_locs = MatchingBBoxes(prior_bboxes, gt_bboxes, gt_labels, cfg)
  local neg_indices = MineHardExamples(conf_preds:float(), match_indices:float())
  local loc_gt_data, loc_pred_data = EncodeLocPrediction(loc_preds:float(), prior_bboxes, gt_locs, match_indices, cfg)
  local conf_gt_data, conf_pred_data = EncodeConfPrediction(conf_preds:float(), match_indices, neg_indices, prior_bboxes:size(), cfg)
  return GetGradient(loc_gt_data, loc_pred_data, conf_gt_data, conf_pred_data, match_indices, neg_indices, prior_bboxes:size(), cfg)
end

function NMS(original_bboxes, original_conf, original_classes, threshold)
  local pick = {}
  local bboxes = original_bboxes:clone()
  local classes = original_classes:clone()
  local sorted_score, i = original_conf:sort(true)
  classes = classes:index(1, i)
  bboxes = bboxes:index(1, i)
  while i:dim() ~= 0 do
    local idx = i[1]
    table.insert(pick, idx)
    local overlaps = JaccardOverlap(bboxes, original_bboxes[idx]):view(-1)
    local diff_bboxes = torch.add(overlaps:lt(threshold), classes:ne(original_classes[idx])):ne(0)
    i = i[diff_bboxes]
    if i:dim() == 0 then
      break
    end
    classes = classes[diff_bboxes]
    local non_zero = diff_bboxes:nonzero()
    bboxes = bboxes:index(1, non_zero:reshape(non_zero:size(1)))
  end
  pick = torch.LongTensor{pick}:reshape(#pick)
  return original_bboxes:index(1, pick), original_classes:index(1, pick), original_conf:index(1, pick)
end

function Detect(model, imgs, nms_threshold, conf_threshold, cfg)
  if imgs:dim() ~= 4 then
    imgs = imgs:reshape(1, imgs:size(1), imgs:size(2), imgs:size(3))
  end
  local outputs = model:forward(imgs:cuda())
  local conf_preds = outputs[2]:float()
  local all_bboxes = {}
  local all_classes = {}
  local all_scores = {}
  local prior_bboxes = GetPriorBBoxes(cfg)
  for i = 1, imgs:size(1) do
    local loc_preds = DecodeBBox(outputs[1][i]:float(), prior_bboxes, cfg.variance)
    local softmax_conf = nn.SoftMax():forward(conf_preds[i]):view(-1,cfg.classes)
    local conf, cls = softmax_conf:narrow(2,2,cfg.classes-1):max(2)
    conf = conf:view(-1)
    local idx = conf:ge(conf_threshold)
    if idx:sum() ~= 0 then
      cls = cls[idx]
      conf = conf[idx]
      local non_zero = idx:nonzero()
      loc_preds = loc_preds:index(1, non_zero:reshape(non_zero:size(1)))
      local bboxes, classes, score = NMS(loc_preds, conf:view(-1), cls:view(-1), nms_threshold)
      bboxes[{{}, {1}}] = torch.cmax(bboxes[{{}, {1}}], 0)
      bboxes[{{}, {2}}] = torch.cmax(bboxes[{{}, {2}}], 0)
      bboxes[{{}, {3}}] = torch.cmin(bboxes[{{}, {3}}], 1)
      bboxes[{{}, {4}}] = torch.cmin(bboxes[{{}, {4}}], 1)
      table.insert(all_bboxes, bboxes)
      table.insert(all_classes, classes)
      table.insert(all_scores, score)
    else
      local dammy_tensor = torch.Tensor{0}
      table.insert(all_bboxes, dammy_tensor)
      table.insert(all_classes, dammy_tensor)
      table.insert(all_scores, dammy_tensor)
    end
  end
  return all_bboxes, all_classes, all_scores
end

function DrawRect(img, box, cls, index2class)
  for i = 1, box:size(1) do
    img = image.drawRect(img, box[i][1], box[i][2], box[i][3], box[i][4])
    img = image.drawText(img, index2class[cls[i]], box[i][1], box[i][2], {color={255,255,255}, bg={0,0,255}, size=1})
  end
  return img
end
