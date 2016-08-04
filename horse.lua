------------------------------------------------------------------
-- Define trainLoader & testLoader, do all the image process here.
------------------------------------------------------------------
require 'image';

dofile('dataloader.lua')

trainDataPath = '/search/ssd/liukuang/cifar10/train/'
trainListPath = '/search/ssd/liukuang/cifar10/train.txt'
testDataPath = '/search/ssd/liukuang/cifar10/test/'
testListPath = '/search/ssd/liukuang/cifar10/test.txt'

-- The default image processing function: just scale the images.
-- It's essential to resize before computing the training mean & std.
imgfunc = function(im) return image.scale(im, 32, 32) end

-- Init trainLoader
if paths.filep('trainLoader.t7') then
    print('loading trainLoader from cache...')
    trainLoader = torch.load('trainLoader.t7')
else
    print('init trainLoader...')
    trainLoader = DataLoader(trainDataPath, trainListPath, imgfunc)
end

-- Init testLoader
if paths.filep('testLoader.t7') then
    print('loading testLoader from cache...')
    testLoader = torch.load('testLoader.t7')
else
    print('init testLoader...')
    testLoader = DataLoader(testDataPath, testListPath, imgfunc)
end

-- Compute training mean & std
if paths.filep('meanstd.t7') then
    print('loading training mean & std from cache...')
    local cache = torch.load('meanstd.t7')
    mean = cache.mean
    std = cache.std
else
    print('computing training mean & std...')
    local nSamples = math.min(10000, trainLoader.nSamples)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for i = 1,nSamples do
        xlua.progress(i,nSamples)
        local im = trainLoader:sample(1)[1]
        for j = 1,3 do
            mean[j] = mean[j] + im[j]:mean()
            std[j] = std[j] + im[j]:std()
        end
    end
    mean:div(nSamples)
    std:div(nSamples)

    local cache = {}
    cache.mean = mean
    cache.std = std
    torch.save('meanstd.t7', cache)
end

function imageProcess(im)
    -------------------------------------------------------------
    -- This is a hook function for processing every loaded image.
    -- Like scale, zero-mean, normalization, augumentation.
    -- We separate it from dataloader for simplicity.
    -------------------------------------------------------------
    -- scale
    im = image.scale(im, 32, 32)
    -- zero-mean & normalization
    for i = 1,3 do  -- for RGB channel
        im[i]:add(-mean[i])
        im[i]:div(std[i])
    end
    return im
end

-- Update the imageProcess function before training
trainLoader.imageProcess = imageProcess
testLoader.imageProcess = imageProcess

-- Save trainLoader & testLoader
if not paths.filep('trainLoader.t7') then
    torch.save('trainLoader.t7', trainLoader)
end

if not paths.filep('testLoader.t7') then
    torch.save('testLoader.t7', testLoader)
end
