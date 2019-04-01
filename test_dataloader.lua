--test dataloader
require 'hdf5';
local utils = require 'fast_neural_style.utils'
require 'fast_neural_style.DataLoader'
local preprocess = require 'fast_neural_style.preprocess'
local cmd = torch.CmdLine()
cmd:option('-h5_file', 'h5Data/data/ms-coco-256.h5');
cmd:option('-batch_size', 2);
cmd:option('-task', 'style', 'style|upsample');
cmd:option('-upsample_factor', 4);
cmd:option('-max_train', -1);
cmd:option('-preprocessing', 'vgg')
opt = cmd:parse(arg)
loader = DataLoader(opt)
x1, x2 = loader:getBatch('train_img')
y1, y2 = loader:getBatch('train_seg')
z1, z2 = loader:getBatch('val_img')
w1, w2 = loader:getBatch('val_img')
return x1, x2, y1,y2, z1,z2,w1,w2