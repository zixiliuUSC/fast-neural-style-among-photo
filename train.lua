require 'torch'
require 'optim'
require 'image'
--在调试时候输入，qlua -lenv
require 'fast_neural_style.DataLoader'
require 'fast_neural_style.PerceptualCriterion'
require 'cutorch'
local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'
local models = require 'fast_neural_style.models'
local extract = require 'fast_neural_style.ExtractMask'
--local breakpoint = require 'breakpoint.active'

local cmd = torch.CmdLine()


--[[
Train a feedforward style transfer model
--]]

-- Generic options
cmd:option('-arch', 'c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3')
cmd:option('-use_instance_norm', 1)
cmd:option('-task', 'style', 'style|upsample')
cmd:option('-h5_file', 'data/ms-coco-256.h5')
cmd:option('-padding_type', 'reflect-start')
cmd:option('-tanh_constant', 150)
cmd:option('-preprocessing', 'vgg')
cmd:option('-resume_from_checkpoint', '')
-- 增加：语义分割模板
-- style_seg是分割模板（是一个mask）

-- Generic loss function options
-- 普通的损失函数
--L1, L2, Smooth L1是指L1范数（绝对值相加），L2是L2范数（平方差除以宽度，再求和）
cmd:option('-pixel_loss_type', 'L2', 'L2|L1|SmoothL1')
cmd:option('-pixel_loss_weight', 0.1)
cmd:option('-percep_loss_weight', 1.0)
--TVLoss,差分正则化
cmd:option('-tv_strength', 1e-6)

-- Options for feature reconstruction loss
-- 内容损失参数
cmd:option('-content_weights', '1.0')
cmd:option('-content_layers', '16')
cmd:option('-loss_network', 'models/vgg16.t7')

-- Options for style reconstruction loss
-- 风格损失参数
cmd:option('-style_image', 'images/styles/style1.png')
cmd:option('-style_image_size', 256)
cmd:option('-style_weights', '5.0')
cmd:option('-style_layers', '4,9,16,23')
cmd:option('-style_target_type', 'gram', 'gram|mean')
-- 这个选项应该是作者后来删的
-- cmd:option('-style_loss_type', 'L2', 'L2|L1|SmoothL1')

-- options for semantic segmentation
cmd:option('-style_seg', 'images/styles/style1_seg.png', 'Style segmentation')
cmd:option('-content_seg', '', 'Content segmentation')

-- Upsampling options
cmd:option('-upsample_factor', 4)

-- Optimization
cmd:option('-num_iterations', 40000)
cmd:option('-max_train', -1)
cmd:option('-batch_size', 4)
cmd:option('-learning_rate', 1e-3)
cmd:option('-lr_decay_every', -1)
cmd:option('-lr_decay_factor', 0.5)
cmd:option('-weight_decay', 0)

-- Checkpointing
cmd:option('-checkpoint_name', 'checkpoint')
cmd:option('-checkpoint_every', 1000)
cmd:option('-num_val_batches', 10)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda', 'cuda|opencl')




function main()
  --理解：parse()函数把cmd中所有的字符都
  --转换为cmdline中的字符串，并为
  --定义每种属性，添加相应的默认值
  
  local opt = cmd:parse(arg)
  
  -- Parse layer strings and weights
  -- 对opt的每种属性的默认值字符串进行分析，并转换为torch中相应的类型
  opt.content_layers, opt.content_weights =
    utils.parse_layers(opt.content_layers, opt.content_weights)
  --utils.parse_layers()对opt.style_layers按','分割，当输入的opt.style_weights只有一个数时，所有层都用同一个权值，否则opt.style_weights数目
  --要与opt.style_layers数目相同
  opt.style_layers, opt.style_weights =
    utils.parse_layers(opt.style_layers, opt.style_weights)

  -- Figure out preprocessing
  -- 指定是vgg的preprocess
  if not preprocess[opt.preprocessing] then
    local msg = 'invalid -preprocessing "%s"; must be "vgg" or "resnet"'
    error(string.format(msg, opt.preprocessing))
  end
  preprocess = preprocess[opt.preprocessing]

  -- Figure out the backend
  -- 指出硬件支持
  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

  -- Build the model
  -- 建立模型
  
  --cutorch.setDevice(1)
  local model = nil
  --查询是否在上次的训练中停止了，如果停止了从上次的继续开始
  if opt.resume_from_checkpoint ~= '' then
    print('Loading checkpoint from ' .. opt.resume_from_checkpoint)
    model = torch.load(opt.resume_from_checkpoint).model:type(dtype)
  else
    print('Initializing model from scratch')
    model = models.build_model(opt):type(dtype)
  end
  --把一般的nn转换为cudnn
  if use_cudnn then cudnn.convert(model, cudnn) end
  model:training()
  print(model)
  
  -- Set up the pixel loss function
  -- 建立像素损失函数
  local pixel_crit
  if opt.pixel_loss_weight > 0 then
    if opt.pixel_loss_type == 'L2' then
      pixel_crit = nn.MSECriterion():type(dtype)
    elseif opt.pixel_loss_type == 'L1' then
      pixel_crit = nn.AbsCriterion():type(dtype)
    elseif opt.pixel_loss_type == 'SmoothL1' then
      --type（dtype）指定类型，由上面的代码可知它可以使用gpu，也可以不用，取决于输入参数
      pixel_crit = nn.SmoothL1Criterion():type(dtype)
    end
  end

  -- Set up the perceptual loss function
  -- 感知损失(即neural style里的Ls)，这里是搭建求算style loss的网络
  --cutorch.setDevice(2)
  local percep_crit
  if opt.percep_loss_weight > 0 then
    local loss_net = torch.load(opt.loss_network)
    local crit_args = {
      cnn = loss_net,
      style_layers = opt.style_layers,
      style_weights = opt.style_weights,
      content_layers = opt.content_layers,
      content_weights = opt.content_weights,
      agg_type = opt.style_target_type,
      --这里还可以设定style_loss的类别，L2或者L1，或L1Smooth, 但作者没给出这样的选择
      --loss_type = opt.style_loss_type,
    }
    percep_crit = nn.PerceptualCriterion(crit_args):type(dtype)
    if opt.task == 'style' then
      -- Load the style image and set it
      -- 用于计算style loss 的网络加载好以后，就调用setStyleTarget求出风格图的特征
      -- torch把图片加载到平台上会自动把图片由0-255转到0-1，线性变换
      local style_image = image.load(opt.style_image, 3, 'float')
      local style_seg = image.load(opt.style_seg, 3, 'float')  --修改【增加：
      style_image = image.scale(style_image, opt.style_image_size)
      style_seg = image.scale(style_seg, opt.style_image_size)
      local H, W = style_image:size(2), style_image:size(3)
      style_image = preprocess.preprocess(style_image:view(1, 3, H, W))
      --修改【增加：
      style_seg = image.scale(style_seg, H, W, 'bilinear')
      local color_codes = {'blue', 'green', 'black', 'white', 'red', 'yellow', 'grey', 'lightblue', 'purple', 'water', 'land', 'mountain'}
      --建筑、物品、背景、汽车、人、植物、公路、天空、船、水、草地、山
      --把style_image_seg分割成9个颜色，存到color_style_masks
      local color_style_masks = {}
      for j = 1, #color_codes do 
        local style_mask_j = extract.ExtractMask(style_seg, color_codes[j])
        table.insert(color_style_masks, style_mask_j)
      end
      style_color={}

      --设定percep_crit，中style_loss_layers的语义分割模板
      percep_crit:setStyleSeg(color_style_masks, opt.style_layers, color_codes)--这里不用担心内容图的分割种类是不是在风格图的分割中，因为在语义分割的时候就做好了限定
      --】
      style_image = style_image:repeatTensor(1,1,1,1)
      percep_crit:setStyleTarget(style_image:type(dtype))
    end
  end
  --cutorch.setDevice(2)
  --cudnn.convert(percep_crit.net, cudnn)
  collectgarbage()
  --print(collectgarbage('count'))
  --print({percep_crit.net})

  local loader = DataLoader(opt)
  local params, grad_params = model:getParameters()
  --shave_y是用来修剪content_target（y）使之与model输出相同大小
  local function shave_y(x, y, out)
    if opt.padding_type == 'none' then
      local H, W = x:size(3), x:size(4)
      local HH, WW = out:size(3), out:size(4)
      local xs = (H - HH) / 2
      local ys = (W - WW) / 2
      return y[{{}, {}, {xs + 1, H - xs}, {ys + 1, W - ys}}]
    else
      return y
    end
  end
  
  -- 注意：！！！f(x)是这里最重要的函数--要优化的函数，用于对model优化的闭包，包含了loss的计算，以及dloss/dinput（64行）
  local function f(x)
    assert(x == params)
    grad_params:zero();

    --!!!这里先当输入图片没有打乱顺序
    --x是输入model的图片; y是content_target, 用于计算loss
    --修改【
    local x, y = loader:getBatch('train_img')
    local pic_num = x:size(1)
    local x_seg, y_seg = torch.Tensor(#x):float()
    --x_seg, y_seg是输入图片的分割模板
    local x_seg_, y_seg_ = loader:getBatch('train_seg')
    --对输入的内容图的语义分割进行拉伸，确保与内容图大小相同
    for i=1, pic_num do 
      x_seg[i] = image.scale(x_seg_[i], x:size(4), x:size(3), 'bilinear')--宽度在前，高度在后
      --y_seg[i] = image.scale(y_seg_[i], y:size(4), y:size(3), 'bilinear')
    end
    --】
    

    --改变数据类型
    x, y = x:type(dtype), y:type(dtype)
    --x_seg = x_seg:type(dtype)

    

    -- Run model forward
    local out = model:forward(x) 
    local grad_out = nil--输出反向传播的梯度

    -- This is a bit of a hack: if we are using reflect-start padding and the
    -- output is not the same size as the input, lazily add reflection padding
    -- to the start of the model so the input and output have the same size.
    --如果是镜像补零则在网络输入加一层，使得out与y大小相同
    if opt.padding_type == 'reflect-start' and x:size(3) ~= out:size(3) then
      local ph = (x:size(3) - out:size(3)) / 2
      local pw = (x:size(4) - out:size(4)) / 2
      local pad_mod = nn.SpatialReflectionPadding(pw, pw, ph, ph):type(dtype)
      model:insert(pad_mod, 1)
      out = model:forward(x)
    end

    --如果是非镜像补零，则强制修剪y使得y与out大小相同
    y = shave_y(x, y, out)
    x_seg = shave_y(x, x_seg, out)

    --修改【
    local color_codes = {'blue', 'green', 'black', 'white', 'red', 'yellow', 'grey', 'lightblue', 'purple', 'water', 'land', 'mountain'}
    --把内容图的语义分割按颜色分离，保存在color_content_masks中
    local color_content_masks={}
    local content_mask_j = nil
    for i = 1, pic_num do 
      local color_content_single = {}
      for j = 1, #color_codes do 
        content_mask_j = extract.ExtractMask(x_seg[i],color_codes[j])
        table.insert(color_content_single, content_mask_j)
      end
      table.insert(color_content_masks, color_content_single)
    end
    --在style_loss层中添加内容图的语义分割模板
    percep_crit:setContentSeg(color_content_masks, opt.style_layers, color_codes)
    --print({percep_crit.net})
    --print(percep_crit.style_loss_layers[1].content_masks)
    -- Compute pixel loss and gradient
    local pixel_loss = 0
    --当'-pixel_loss_type'有合法输入时，pixel_crit才会有网络，才不会是nil，这时下面的if才会执行，pixel_loss才不为0
    if pixel_crit then
      --因为pixel_crit整个网络的输出是loss，所以直接赋值到pixel_loss（详见train.lua的125到134行），
      pixel_loss = pixel_crit:forward(out, y)
      pixel_loss = pixel_loss * opt.pixel_loss_weight
      local grad_out_pix = pixel_crit:backward(out, y)
      --函数f(x)要执行很多次（很多个batch），只有第一次执行时grad_out才是nil 
      if grad_out then
        --当grad_out不是nil时，grad_out = grad_out + pixel_loss_weight x grad_out_pix
        grad_out:add(opt.pixel_loss_weight, grad_out_pix)
      else
        --当grad_out是nil时，grad_out = pixel_loss_weight x grad_out_pix
        grad_out_pix:mul(opt.pixel_loss_weight)
        grad_out = grad_out_pix
      end
    end

    -- Compute perceptual loss and gradient
    -- 修改【对content的语义分割进行拆分

    local percep_loss = 0
    local target = nil
    local grad_out_percep = nil
    if percep_crit then
      target = {content_target=y}--即 target.content_target=y
      --这里同理，percep_crit的网络输出是loss（详见perceptualcriterion.lua的156行）
      --一句forward即算出percep_loss，要改的东西在percep_crit的类文件PerceptualCriterion.lua
      percep_loss = percep_crit:forward(out, target)
      percep_loss = percep_loss * opt.percep_loss_weight
      grad_out_percep = percep_crit:backward(out, target)
      
      --函数f(x)要执行很多次（很多个batch），只有第一次执行时grad_out才是nil
      if grad_out then
        --当grad_out不是nil时，grad_out = grad_out + percep_loss_weight x grad_out_percep
        grad_out:add(opt.percep_loss_weight, grad_out_percep)
      else
        --当grad_out是nil时，grad_out = percep_loss_weight x grad_out_percep
        grad_out_percep:mul(opt.percep_loss_weight)
        grad_out = grad_out_percep
      end  
    end

    local loss = pixel_loss + percep_loss
    --breakpoint('check loss net memory')
    -- Run model backward
    model:backward(x, grad_out)
    --breakpoint('check loss net memory')
    -- Add regularization
    -- grad_params:add(opt.weight_decay, params)
 	  collectgarbage()
    --print(collectgarbage('count'))
    return loss, grad_params
  end


  local optim_state = {learningRate=opt.learning_rate}
  local train_loss_history = {}
  local val_loss_history = {}
  local val_loss_history_ts = {}
  local style_loss_history = nil
  if opt.task == 'style' then
    style_loss_history = {}
    for i, k in ipairs(opt.style_layers) do
      style_loss_history[string.format('style-%d', k)] = {}
    end
    for i, k in ipairs(opt.content_layers) do
      style_loss_history[string.format('content-%d', k)] = {}
    end
  end

  local style_weight = opt.style_weight
  for t = 1, opt.num_iterations do
    local epoch = t / loader.num_minibatches['train_img']

    --调用optim进行优化
    local _, loss = optim.adam(f, params, optim_state)

    table.insert(train_loss_history, loss[1])

    if opt.task == 'style' then
      for i, k in ipairs(opt.style_layers) do
        table.insert(style_loss_history[string.format('style-%d', k)],
          percep_crit.style_losses[i])
      end
      for i, k in ipairs(opt.content_layers) do
        table.insert(style_loss_history[string.format('content-%d', k)],
          percep_crit.content_losses[i])
      end
    end

    print(string.format('Epoch %f, Iteration %d / %d, loss = %f',
          epoch, t, opt.num_iterations, loss[1]), optim_state.learningRate)
    local getTime = os.date('%c')
    print(getTime)
    --breakpoint('check video memory')

    
    if t % opt.checkpoint_every == 0 then
      -- Check loss on the validation set
      --[[
      loader:reset('val')
      model:evaluate()
      local val_loss = 0
      print 'Running on validation set ... '
      local val_batches = opt.num_val_batches
      for j = 1, val_batches do
        local x, y = loader:getBatch('val_img')
        x, y = x:type(dtype), y:type(dtype)
        local out = model:forward(x)
        y = shave_y(x, y, out)
        local pixel_loss = 0
        if pixel_crit then
          pixel_loss = pixel_crit:forward(out, y)
          pixel_loss = opt.pixel_loss_weight * pixel_loss
        end
        local percep_loss = 0
        if percep_crit then
          percep_loss = percep_crit:forward(out, {content_target=y})
          percep_loss = opt.percep_loss_weight * percep_loss
        end
        val_loss = val_loss + pixel_loss + percep_loss
      end
      val_loss = val_loss / val_batches
      print(string.format('val loss = %f', val_loss))
      table.insert(val_loss_history, val_loss)
      table.insert(val_loss_history_ts, t)
      model:training()
	    --]]

      -- Save a JSON checkpoint
      local checkpoint = {
        opt=opt,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_loss_history_ts=val_loss_history_ts,
        style_loss_history=style_loss_history,
      }
      local filename = string.format('%s.json', opt.checkpoint_name)
      paths.mkdir(paths.dirname(filename))
      utils.write_json(filename, checkpoint)

      -- Save a torch checkpoint; convert the model to float first
      model:clearState()
      if use_cudnn then
        cudnn.convert(model, nn)
      end
      model:float()
      checkpoint.model = model
      filename = string.format('%s.t7', opt.checkpoint_name)
      torch.save(filename, checkpoint)

      -- Convert the model back
      model:type(dtype)
      if use_cudnn then
        cudnn.convert(model, cudnn)
      end
      params, grad_params = model:getParameters()
    end

    if opt.lr_decay_every > 0 and t % opt.lr_decay_every == 0 then
      local new_lr = opt.lr_decay_factor * optim_state.learningRate
      optim_state = {learningRate = new_lr}
    end
  end
end

--[[
function ExtractMask(seg, color)
  local mask = nil
  if color == 'green' then 
    mask = torch.lt(seg[1], 0.1)
    mask:cmul(torch.gt(seg[2], 1-0.1))
    mask:cmul(torch.lt(seg[3], 0.1))
  elseif color == 'black' then 
    mask = torch.lt(seg[1], 0.1)
    mask:cmul(torch.lt(seg[2], 0.1))
    mask:cmul(torch.lt(seg[3], 0.1))
  elseif color == 'white' then
    mask = torch.gt(seg[1], 1-0.1)
    mask:cmul(torch.gt(seg[2], 1-0.1))
    mask:cmul(torch.gt(seg[3], 1-0.1))
  elseif color == 'red' then 
    mask = torch.gt(seg[1], 1-0.1)
    mask:cmul(torch.lt(seg[2], 0.1))
    mask:cmul(torch.lt(seg[3], 0.1))
  elseif color == 'blue' then
    mask = torch.lt(seg[1], 0.1)
    mask:cmul(torch.lt(seg[2], 0.1))
    mask:cmul(torch.gt(seg[3], 1-0.1))
  elseif color == 'yellow' then
    mask = torch.gt(seg[1], 1-0.1)
    mask:cmul(torch.gt(seg[2], 1-0.1))
    mask:cmul(torch.lt(seg[3], 0.1))
  elseif color == 'grey' then 
    mask = torch.cmul(torch.gt(seg[1], 0.5-0.1), torch.lt(seg[1], 0.5+0.1))
    mask:cmul(torch.cmul(torch.gt(seg[2], 0.5-0.1), torch.lt(seg[2], 0.5+0.1)))
    mask:cmul(torch.cmul(torch.gt(seg[3], 0.5-0.1), torch.lt(seg[3], 0.5+0.1)))
  elseif color == 'lightblue' then
    mask = torch.lt(seg[1], 0.1)
    mask:cmul(torch.gt(seg[2], 1-0.1))
    mask:cmul(torch.gt(seg[3], 1-0.1))
  elseif color == 'purple' then 
    mask = torch.gt(seg[1], 1-0.1)
    mask:cmul(torch.lt(seg[2], 0.1))
    mask:cmul(torch.gt(seg[3], 1-0.1))
  else 
    print('ExtractMask(): color not recognized, color = ', color)
  end 
  return mask:float():cuda()
end
--]]
main()

