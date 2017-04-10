model_path = 'model120000iteration.t7'
img_path ='path/to/image'
_=dofile('ssd.llua')
dofile('cfg.lua')
dofile('utils.lua')
require 'image'

model = torch.load(model_path):cuda()
img = image.load(img_path)
res = img.clone()

class_list = {'__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
              'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}

img_r = img[{{1},{},{}}]
img_b = img[{{3},{},{}}]
img[{{1},{},{}}] = img_b
img[{{3},{},{}}] = img_r
boxes, classes, scores = Detect(model, img, 0.45, 0.6, cfg)
res = DrawRect(res, boxes[1], classes[1], class_list)

image.savePNG('res.png', res)
