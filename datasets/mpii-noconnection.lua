--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  MPII dataset loader (from  (Newell))
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/posetransforms'


-------------------------------------------------------------------------------
-- Helper Functions
-------------------------------------------------------------------------------
local getTransform = t.getTransform
local transform = t.transform
local crop = t.crop2
local drawGaussian = t.drawGaussian
local drawresult = t.drawresult
local drawMask = t.drawMask
local drawHead= t.drawHead
local shuffleLR = t.shuffleLR
local flip = t.flip
local colorNormalize = t.colorNormalize

-------------------------------------------------------------------------------
-- Create dataset Class
-------------------------------------------------------------------------------

local M = {}
local MpiiDataset = torch.class('resnet.MpiiDataset', M)

function MpiiDataset:__init(imageInfo, opt, split)
  assert(imageInfo[split], split)
  self.imageInfo = imageInfo[split]--比如split，那么就是对应train那个维度下面的东西
  self.split = split
  -- Some arguments
  self.inputRes = opt.inputRes
  self.outputRes = opt.outputRes
  -- Options for augmentation
  self.scaleFactor = opt.scaleFactor
  self.rotFactor = opt.rotFactor
  self.dataset = opt.dataset
  self.nStack = opt.nStack
  self.meanstd = torch.load('gen/mpii/meanstd.t7')
  self.nGPU = opt.nGPU
  self.batchSize = opt.batchSize
  self.minusMean = opt.minusMean
  self.gsize = opt.gsize
  self.bg = opt.bg
  self.rotProbab = opt.rotProbab
end

function MpiiDataset:get(i, scaleFactor)
   local scaleFactor = scaleFactor or 1
   local img = image.load(paths.concat('data/mpii/images', self.imageInfo.data['images'][i]))
   local imgname=self.imageInfo.data['images'][i]
   -- Generate samples
   local pts = self.imageInfo.labels['part'][i]--对应一个图像中的样本数量
   local c = self.imageInfo.labels['center'][i]
   local s = self.imageInfo.labels['scale'][i]*scaleFactor

   -- For single-person pose estimation with a centered/scaled figure
   local nParts = pts:size(1)
   local inp = crop(img, c, s, 0, self.inputRes)--对图像进行裁剪
   --local out = self.bg == 'true' and torch.zeros((nParts+1)*2, self.outputRes, self.outputRes)
                               -- or torch.zeros((nParts)*2, self.outputRes, self.outputRes)
   --local temp_mask = self.bg == 'true' and torch.zeros((nParts+1)*2, self.outputRes, self.outputRes)
                               -- or torch.zeros((nParts)*2, self.outputRes, self.outputRes)

   local testidx=i
   local out = torch.zeros(nParts+1, self.outputRes, self.outputRes)
   local temp_mask =  torch.zeros(nParts, self.outputRes, self.outputRes)
   local scale_pt=torch.zeros(nParts,2)
   local target_out = torch.zeros(nParts, self.outputRes, self.outputRes)

   for i = 1,nParts do
     if pts[i][1] > 0 then --
         scale_pt[i]=transform(torch.add(pts[i],1), c, s, 0, self.outputRes)--获取转换后坐标点
         drawGaussian(out[i], scale_pt[i], self.gsize)--绘制高斯点
         --image.save('show/'..imgname.."热图"..i.."--"..testidx.."--"..imgname, out[i])
        --print(out[i]:size())
        --if  i~=10  then--头顶不生成mask
           drawMask(temp_mask[i], scale_pt[i], self.gsize)--绘制关节点
        -- end
        --image.save('show/'..imgname.."掩码"..i.."--"..testidx.."--"..imgname, temp_mask[i])
     end--对输出图像绘制高斯图
   end

   --image.save('show/'..imgname.."--"..testidx.."--"..imgname, inp)
   --print("热图前坐标点为:x1:"..scale_pt[1][1]..'-Y1:'..scale_pt[1][2]..'-X2:'..scale_pt[2][1]..'-Y2:'..scale_pt[2][2])
   local mask=drawresult(temp_mask,nParts,scale_pt)--生成预测热图,并且将关节点连接起来，这里没有将关节点连接起来
   --drawHead(mask,scale_pt[9],scale_pt[10])
   out[nParts+1]=mask--这样才能保证数据增强时变换一致
   --image.save('show/'..imgname.."mymask".."--"..testidx.."--"..imgname,mask)
   --print("一张训练label产生完毕")


   -- Data augmentation
   inp, out = self.augmentation(self, inp, out)
   collectgarbage()
   mask=out[nParts+1]

   for i=1 , nParts do
       target_out[i]=out[i]
   end
   local target_mask =  torch.zeros(1, self.outputRes, self.outputRes)
    target_mask[1]=mask
   return {
      input = inp,
      target =target_out,
      target_mask=target_mask,--temp_mask,
      center = c,
      scale = s,
      width = img:size(3),
      height = img:size(2),
      imgPath = paths.concat('data/mpii/images', self.imageInfo.data['images'][i])
   }
end

function MpiiDataset:size()
  if self.split == 'test' then
     --return 240
     return self.imageInfo.labels.nsamples
  end
  
   local nSamples = self.imageInfo.labels.nsamples - (self.imageInfo.labels.nsamples%self.nGPU)--nGPU就是1啦
   nSamples = nSamples - nSamples%self.batchSize--返回的是样本数量吧，但是这里nSamples%self.batchSize依然是0
    --imageInfo.labels.nsamples等于nsample,也就是对应的part字段

   return nSamples
   --return 240
end

function MpiiDataset:preprocess()
   return function(img)
      if img:max() > 2 then
         img:div(255)
      end
      return self.minusMean == 'true' and colorNormalize(img, self.meanstd) or img
   end
end

function MpiiDataset:augmentation(input, label)
  -- Augment data (during training only)
  if self.split == 'train' then
      local s = torch.randn(1):mul(self.scaleFactor):add(1):clamp(1-self.scaleFactor,1+self.scaleFactor)[1]
      local r = torch.randn(1):mul(self.rotFactor):clamp(-2*self.rotFactor,2*self.rotFactor)[1]

      -- Color
      input[{1, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      input[{2, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      input[{3, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)

      -- Scale/rotation
      if torch.uniform() <= (1-self.rotProbab) then r = 0 end
      local inp,out = self.inputRes, self.outputRes
      input = crop(input, torch.Tensor({(inp+1)/2,(inp+1)/2}), inp*s/200, r, inp)
      label = crop(label, torch.Tensor({(out+1)/2,(out+1)/2}), out*s/200, r, out)

      -- Flip
      if torch.uniform() <= .5 then
          input = flip(input)
          label = flip(shuffleLR(label, self.dataset))
      end
  end

  return input, label
end

return M.MpiiDataset
