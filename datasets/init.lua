--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet and CIFAR-10 datasets
--

local M = {}

local function isvalid(opt, cachePath)
   local imageInfo = torch.load(cachePath)
   if imageInfo.basedir and imageInfo.basedir ~= opt.data then
      return false
   end
   return true
end

function M.create(opt, split)--opt：操作命令集合，split：'train', 'val', 'test'
   local cachePath = paths.concat(opt.gen, opt.dataset .. '.t7')--opt.gen应该是文件路径名称，这个东西指定了图片文件和标签文件最终保存路径
   if not paths.filep(cachePath) or not isvalid(opt, cachePath) then
      paths.mkdir('gen')

      local script = paths.dofile(opt.dataset .. '-gen.lua')--执行对应文件路径下面的函数
      script.exec(opt, cachePath)--执行那个脚本对应下面的函数,这个函数是用于生成对应的t7文件
   end
   local imageInfo = torch.load(cachePath)--data和label全部都被加载起来啦，从t7文件里面读出来的

   local Dataset = require('datasets/' .. opt.dataset)--调用对应的MPII.lua
   return Dataset(imageInfo, opt, split)--返回一个dataset/mpii.lua下面的MpiiDataset:__init对象
end

return M
