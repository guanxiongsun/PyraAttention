#!/usr/bin/env sh
expID=mpii/hg-prm-stack2 #快照和日志文件将保存在checkpoints/$expID
dataset=mpii
gpuID=0 #编程所用GPU
nGPU=1 #所使用的GPU数量
batchSize=6
LR=2.5e-4
netType=hg-prm #所使用的网络架构
nStack=6
nResidual=1
nThreads=8	# 使用多少个线程来加载数据
minusMean=true
nClasses=16
nEpochs=250
snapshot=10
nFeats=256
baseWidth=9
cardinality=4
#这是一个训练的demo实例，采用了2层堆叠的模型
CUDA_VISIBLE_DEVICES=$gpuID th main.lua \
	-dataset $dataset \
	-expID $expID \
	-batchSize $batchSize \
	-nGPU $nGPU \
	-LR $LR \
	-momentum 0.0 \
	-weightDecay 0.0 \
	-netType $netType \
	-nStack $nStack \
	-nResidual $nResidual \
	-nThreads $nThreads \
	-minusMean $minusMean \
	-nClasses $nClasses \
	-nEpochs $nEpochs \
	-snapshot $snapshot \
	-nFeats $nFeats \
	-baseWidth $baseWidth \
	-cardinality $cardinality \
	# -resume checkpoints/$expID
