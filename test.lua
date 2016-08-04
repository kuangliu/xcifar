dofile('listdataset.lua')

p = ListDataset({
    trainData = '/search/ssd/liukuang/cifar10/train/',
    trainList = '/search/ssd/liukuang/cifar10/train.txt',
    testData = '/search/ssd/liukuang/cifar10/test/',
    testList = '/search/ssd/liukuang/cifar10/test.txt',
    imsize = 32,
    imfunc = nil -- image processing function, default: zeromean & normalization
})

torch.save('p.t7',p)
