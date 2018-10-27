--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'xlua'
local DataLoader = require 'dataloader'
local models = require 'models.init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
local Logger = require 'utils.Logger'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)--设置并行数量

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)--为CPU设置种子用于生成随机数，以使得结果是确定的 
cutorch.manualSeedAll(opt.manualSeed)

-- 获取以前的保存点文件，初始训练时直接返回nil
local checkpoint, optimState = checkpoints.latest(opt)

-- 建立网络和定义损失函数
local model, criterion = models.setup(opt, checkpoint)


-- 根据参数定义loader,返回三个对象
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)--一些模型的优化，例如学习率啥的，和训练集之类的无关

if opt.testRelease then
   print('=> Test Release')
   local testAcc, testLoss = trainer:test(opt.epochNumber, testLoader)
   print(string.format(' * Results acc: %6.3f, loss: %6.3f', testAcc, testLoss))
   return
end

if opt.testOnly then
   print('=> Test Only')
   local testAcc, testLoss = trainer:test(opt.epochNumber, valLoader)
   print(string.format(' * Results acc: %6.3f, loss: %6.3f', testAcc, testLoss))
   return
end

local r_step, d_step = 3/opt.nEpochs, 5/opt.nEpochs

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestAcc = -math.huge
local bestEpoch = 0
local logger = Logger(paths.concat(opt.save, opt.expID, 'full.log'), opt.resume ~= 'none')
logger:setNames{'Train acc.', 'Train loss.', 'Test acc.', 'Test loss.'}
logger:style{'+-', '+-', '+-', '+-'}
for epoch = startEpoch, opt.nEpochs do

   -- Train for a single epoch
   local trainAcc, trainLoss = trainer:train(epoch, trainLoader)

   local testAcc, testLoss = trainer:test(epoch, valLoader)

   -- Write to logger
   logger:add{trainAcc, trainLoss, testAcc, testLoss}
   print((' Finished epoch # %d'):format(epoch))
   print(('\tTrain Loss: %.4f, Train Acc: %.4f'):format(trainLoss, trainAcc))
   print(('\tTest Loss:  %.4f, Test Acc:  %.4f'):format(testLoss, testAcc))

   local bestModel = false
   if testAcc > bestAcc then
      bestModel = true
      bestAcc = testAcc
      bestEpoch = epoch
      checkpoints.saveBest(epoch, model, opt)
      print(('\tBest model: %.4f [*]'):format(bestAcc))
   end

   if epoch % opt.snapshot == 0 then
      checkpoints.save(epoch, model, trainer.optimState, opt)
   end

   collectgarbage()
end

print(string.format(' * Finished acc: %6.3f, Best epoch: %d', bestAcc, bestEpoch))
