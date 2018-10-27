--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Copyright (c) 2016, YANG Wei
--  Script to prepare MPII dataset
--  The annotations are adopted from: https://github.com/anewell/pose-hg-train/tree/master/data/mpii/annot

local hdf5 = require 'hdf5'

local M = {}

local function convertMPII(file, namelist)
   local data, labels = {}, {}
   local a = hdf5.open(file, 'r')
   local namesFile = io.open(namelist, 'r')


   -- Read in annotation information
   local tags = {'part', 'center', 'scale', 'normalize', 'torsoangle', 'visible'}
   for _,tag in ipairs(tags) do 
      labels[tag] = a:read(tag):all()
      --print(a:read(tag):all())
   end
   labels['nsamples'] = labels['part']:size()[1]--nsmaples等于那个t7文件的part字段个数

   -- Load in image file names (reading strings wasn't working from hdf5)
   data['images'] = {}
   local toIdxs = {}
   local idx = 1
   for line in namesFile:lines() do
     data['images'][idx] = line--data[images]里面保存的全是图像名称，而且一行可以有多个图像
     --print(line.."下标"..idx)
     if not toIdxs[line] then toIdxs[line] = {} end
     table.insert(toIdxs[line], idx)--这个相当于倒排索引啦，下标是图像名称，值则是其出现在行位置的列表
     idx = idx + 1
   end
   namesFile:close()

   -- This allows us to reference multiple people who are in the same image
   data['imageToIdxs'] = toIdxs

   return {
      data = data,--images是正排索引，imagetoIdx是倒排索引
      labels = labels,--读取出来的注释
   }
end

local function computeMean( train )--计算训练集所有图像的RGB通道对应均值，并且保存在gen/mpii/meanstd.t7下面
   print('==> Compute image mean and std')
   local meanstd
   local data, labels = train.data, train.labels
   if not paths.dirp('gen/mpii') then
      paths.mkdir('gen/mpii')
   end   
   if not paths.filep('gen/mpii/meanstd.t7') then
      local size_ = labels['nsamples']
      local rgbs = torch.Tensor(3, size_)

      for idx = 1, size_ do
         xlua.progress(idx, size_)--设置进度条
         local imgpath = paths.concat('data/mpii/images', data['images'][idx])--取出对应id的图像，注意同一图像可能多次使用
         local imdata = image.load(imgpath)--读取图像
         rgbs:select(2, idx):copy(imdata:view(3, -1):mean(2))--select是将第2维的idx取出来，view(x.size(0), -1) 是将tensor转换成size[0]行，mean是指在第二个维度求平均，也就是每个通道的平均，并且放入到rgbs的第二个维度第idx个元素
      end

      local mean = rgbs:mean(2):squeeze()--压缩所有维度为1的维度
      local std = rgbs:std(2):squeeze()

      meanstd = {
         mean = mean,
         std = std
      }

      torch.save('gen/mpii/meanstd.t7', meanstd)--将归一化后的图像保存
   else
      meanstd = torch.load('gen/mpii/meanstd.t7')
   end

   print(('    mean: %.4f %.4f %.4f'):format(meanstd.mean[1], meanstd.mean[2], meanstd.mean[3]))
   print(('     std: %.4f %.4f %.4f'):format(meanstd.std[1], meanstd.std[2], meanstd.std[3]))
   return meanstd
end

function M.exec(opt, cacheFile)
   print(" 开始建立数据集！ " )
   local trainData = convertMPII('data/mpii/train.h5', 'data/mpii/train_images.txt')
   local validData = convertMPII('data/mpii/valid.h5', 'data/mpii/valid_images.txt')
   local testData = convertMPII('data/mpii/test.h5', 'data/mpii/test_images.txt')

   -- Compute image mean
   computeMean(trainData)

   print(" | saving MPII dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = validData,
      test = testData,
   })--将打包后的data和label保存起来,成为一个t7文件
   print("  train: ".. trainData.labels.nsamples)--样本数量都是对应的标签数量
   print("  valid: ".. validData.labels.nsamples)
   print("  test: ".. testData.labels.nsamples)
end

return M
