_=dofile('ssd.lua')
dofile('cfg.lua')
dofile('utils.lua')
require 'image';
model=torch.load('output/model120000iter.t7')
path =torch.load('cache/paths2007Test.t7')
class_list = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
              'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}
torch.setdefaulttensortype('torch.FloatTensor')
img = image.load('data/VOCdevkit/VOC'..path[1][1]..'/JPEGImages/'..path[1][2]..'.jpg')
img_r = img[{{1},{},{}}]
img_b = img[{{3},{},{}}]
img[{{1},{},{}}] = img_b
img[{{3},{},{}}] = img_r
img = image.scale(img,300,300)
box,cls,score=Detect(model,img,0.45,0.6,cfg)
img = image.load('data/VOCdevkit/VOC'..path[1][1]..'/JPEGImages/'..path[1][2]..'.jpg')
box = box[1]
for i = 1, box:size(1) do
  box[i][1] = box[i][1]*img:size(3)
  box[i][2] = box[i][2]*img:size(2)
  box[i][3] = box[i][3]*img:size(3)
  box[i][4] = box[i][4]*img:size(2)
end
img = DrawRect(img,box,cls[1],class_list)
image.savePNG('test4.png',img)
