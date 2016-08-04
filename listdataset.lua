--------------------------------------------------------------------------------
-- listdataset loads training/test data with list files containing the names
-- and targets.
--------------------------------------------------------------------------------
dofile('dataloader.lua')

local ListDataset = torch.class 'ListDataset'

function ListDataset:__init(args)
    -- parse args
    local argnames = {'trainData', 'trainList',
                      'testData', 'testList',
                      'imsize', 'imfunc'}
    for _,v in pairs(argnames) do
        self[v] = args[v]
    end

    -- init trainloader & testLoder
    self:__initLoader('trainLoader', self.trainData, self.trainList)
    self:__initLoader('testLoader', self.testData, self.testList)
    self:__calcMeanStd() -- compute trainLoader mean & std

    self.ntrain = self.trainLoader.nSamples
    self.ntest = self.testLoader.nSamples

    -- update imfunc, perform zero mean & normalization
    local zmfunc = function(im)
        -- resize
        im = image.scale(im, self.imsize, self.imsize)
        -- zero-mean & normalization
        for i = 1,3 do  -- for RGB channel
            im[i]:add(-self.mean[i])
            im[i]:div(self.std[i])
        end
        return im
    end

    self.trainLoader.__imfunc = args.imfunc or zmfunc
    self.testLoader.__imfunc = args.imfunc or zmfunc

    -- save
    if not paths.filep('./data/trainLoader.t7') then
        torch.save('./data/trainLoader.t7', self.trainLoader)
    end

    if not paths.filep('./data/testLoader.t7') then
        torch.save('./data/testLoader.t7', self.testLoader)
    end
end

---------------------------------------------------------------
-- init trainLoader & testLoader
--
function ListDataset:__initLoader(loaderName, dataPath, listPath)
    if not paths.dirp('data') then os.execute('mkdir data') end

    -- set the default image processing function to resize
    local imscale = function(im) return image.scale(im, self.imsize, self.imsize) end

    local filePath = './data/'..loaderName..'.t7'
    if paths.filep(filePath) then
        print('==> loading '..loaderName..' from cache...')
        loader = torch.load(filePath)
    else
        print('==> init '..loaderName..'...')
        loader = DataLoader(dataPath, listPath, imscale)
    end
    self[loaderName] = loader
end

---------------------------------------------------------------
-- calculate training mean & std
--
function ListDataset:__calcMeanStd()
    local filePath = './data/meanstd.t7'
    local cache, mean, std
    if paths.filep(filePath) then
        print('==> loading mean & std from cache...')
        cache = torch.load(filePath)
        mean = cache.mean
        std = cache.std
    else
        print('==> computing mean & std...')
        local nSamples = math.min(10000, self.trainLoader.nSamples)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for i = 1,nSamples do
            xlua.progress(i,nSamples)
            local im = self.trainLoader:sample(1)[1]
            for j = 1,3 do
                mean[j] = mean[j] + im[j]:mean()
                std[j] = std[j] + im[j]:std()
            end
        end
        cache = {}
        cache.mean = mean:div(nSamples)
        cache.std = std:div(nSamples)
        torch.save(filePath, cache)
    end
    self.mean = mean
    self.std = std
end

---------------------------------------------------------------
-- load training batch sample
--
function ListDataset:sample(quantity)
    return self.trainLoader:sample(quantity)
end

---------------------------------------------------------------
-- load test batch sample
--
function ListDataset:get(i1,i2)
    return self.testLoader:get(i1,i2)
end
