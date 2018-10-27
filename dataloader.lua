--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   sets = {'train', 'val'}
   if opt.testRelease then sets = {'train', 'val', 'test'} end

   for i, split in ipairs(sets) do
      local dataset = datasets.create(opt, split)--datasets/init,获得了mpii.lua对象，用于后续读取操作
      --其方法是将data名称和对应标记生成t7文件，然后使用读取他，生成一个dataset/mpii.lua下面的MpiiDataset:__init对象
      print("changshidiaoyong")
      loaders[i] = M.DataLoader(dataset, opt, split)--产生三个Dataloader类，对应的是训练集，验证集和测试集
   end

   return table.unpack(loaders)--以表格形式返回这三个对象，这些对象已经包含了datainfo的列表，相应字段按照mpii gen里面所示
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   --print("设置随机数种子")
   local function init()
      --print("init")
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      --print("main")
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset--返回一个mpii.lua下面对应MpiiDataset对象
      _G.preprocess = dataset:preprocess()--利用前面取出的均值进行归一化
      return dataset:size()--注意这一个size对应的是MpiiDataset:size(),也就是和nsample相同的
   end
   print("准备启动线程")--这里会先执行init函数
   local threads, sizes = Threads(opt.nThreads, init, main)--并行线程数量
   self.threads = threads
   self.__size = sizes[1][1]--对应的是nsample把
   self.batchSize = opt.batchSize
   self.inputRes = opt.inputRes
   self.outputRes = opt.outputRes
   self.nStack = opt.nStack
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run(randPerm_)
   print("启动线程")
   local randPerm = true
   if randPerm_ ~= nil then
      randPerm = randPerm_
   end
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize--batchsize受到
   local perm = torch.randperm(size)
   if not randPerm then
      perm = torch.range(1, size)
   end--产生一个顺序序列

   local idx, sample = 1, nil
   local nStack = self.nStack--堆叠的

   local function enqueue()
      while idx <= size and threads:acceptsjob() do--idx从1开始
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))--dimension*, *start*, *length
         --print("打印一下瞅瞅")
         --print(indices)--取出来下标列表
         threads:addjob(
            function(indices)
               local sz = indices:size(1)
               local batch, target,mask = nil, nil,nil
               local scale, offset, center, index = {}, {}, {}, {}-- for testing pose 
               
               for i, idx in ipairs(indices:totable()) do
                  --print(idx)
                  local sample = _G.dataset:get(idx)--根据idx去mpii.lua下面去MpiiDataset:getimage
                  local input = _G.preprocess(sample.input)--对采集到的图像进行归一化

                  -- local sample = {}
                  -- sample.input = torch.Tensor(3, 256, 256)
                  -- sample.target = torch.Tensor(16, 64, 64)
                  -- local input = sample.input
                  --[[print('打包前')
                  print("input:")
                  print(input:size())
                  print("target:")
                  print(sample.target[1]:size())
                  print("mask:")
                  print(sample.target_mask[1]:size())--]]
                  if not batch then 
                     batch = input:view(1,unpack(input:size():totable())) 
                  else 
                     batch = batch:cat(input:view(1,unpack(input:size():totable())),1) 
                  end

                  if not target then 
                     target = sample.target:view(1,unpack(sample.target:size():totable())) 
                  else 
                     target = target:cat(sample.target:view(1,unpack(sample.target:size():totable())),1) 
                  end

                  if not mask then
                     mask = sample.target_mask:view(1,unpack(sample.target_mask:size():totable()))
                  else
                     mask = mask:cat(sample.target_mask:view(1,unpack(sample.target_mask:size():totable())),1)
                  end
                  --print('打包后')
                 -- print("batch:")
                  --print(batch:size())
                  --print("target:")
                  --print(target:size())--
                  --print("mask:")
                  --print(mask:size())
                  -- for testing pose
                  scale[i] = sample.scale
                  offset[i] = sample.offset
                  center[i] = sample.center
                  index[i] = idx
                  -- label[i] = sample.label or -1
               end

               -- Set up label for intermediate supervision 这里将batch处理为表格啦
               if nStack > 1 then
                  local targetTable = {}
                  for s = 1, nStack do table.insert(targetTable, target) end
                  target = targetTable

                  local maskTable = {}
                  for s = 1, nStack do table.insert(maskTable, mask) end
                  mask = maskTable

               end
               collectgarbage()
               return {
                  input = batch,
                  target = target,
                  mask=mask,
                  scale = scale,  -- for testing pose
                  offset = offset,  -- for testing pose
                  center = center, -- for testing pose
                  index = index,
                  -- label = label,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices
         )
         idx = idx + batchSize--从数组里面截取更多的元素
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      --print("开始采样")
      enqueue()
      n = n + 1

      --print(n)--n的值是不断递增的
      --print("采样结束")
      --print(#sample)
      --if #sample>1 then
         --print(sample[1].size())

      --end--为啥始终大小为0啊
      return n, sample
   end

   return loop
end

return M.DataLoader
