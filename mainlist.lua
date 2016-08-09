require 'pl'
require 'nn'
require 'nnx'
require 'torch'
require 'optim'
require 'image'
require 'paths'

utils = dofile('utils.lua')
xtorch = dofile('xtorch.lua')

------------------------------------------------
-- 1. prepare data
--
dofile('listdataset.lua')
ds = ListDataset({
    trainData = '/search/ssd/liukuang/cifar10/train/',
    trainList = '/search/ssd/liukuang/cifar10/train.txt',
    testData = '/search/ssd/liukuang/cifar10/test/',
    testList = '/search/ssd/liukuang/cifar10/test.txt',
    imsize = 32
})

------------------------------------------------
-- 2. define net
--
dofile('resnet.lua')
dofile('augment.lua')
net = getResNet()

------------------------------------------------
-- 3. init optimization params
--
optimState = {
    learningRate = 0.01,
    learningRateDecay = 1e-7,
    weightDecay = 1e-4,
    nesterov = true,
    momentum = 0.9,
    dampening = 0.0
}

opt = {
    ----------- net options --------------------
    net = net,
    ----------- data options -------------------
    dataset = ds,
    nhorse = 4,         -- nb of threads to load data, default 2
    ----------- training options ---------------
    batchSize = 128,
    nEpoch = 500,
    nClass = 10,
    ----------- optimization options -----------
    optimizer = optim.sgd,
    criterion = nn.CrossEntropyCriterion,
    optimState = optimState,
    ----------- general options ----------------
    backend = 'GPU',    -- CPU or GPU, default CPU
    nGPU = 4,           -- nb of GPUs to use, default 1
    verbose = true
}

------------------------------------------------
-- 4. and fit!
--
xtorch.fit(opt)
