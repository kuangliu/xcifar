local xtorch = {}

function xtorch.fit(opt)
    xtorch.init(opt)

    for i = 1, opt.nEpoch do
        xtorch.train(opt)
        xtorch.test(opt)
    end
end

----------------------------------------------------------------
-- init global params
--
function xtorch.init(opt)
    net = utils.MSRinit(opt.net)
    criterion = nn.CrossEntropyCriterion()

    -- use GPU
    if opt.backend == 'GPU' then
        require 'cunn'
        require 'cutorch'
        cudnn = require 'cudnn'

        -- insert a copy layer in the first place
        net:insert(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'), 1)

        cudnn.convert(net, cudnn):cuda()
        cudnn.fastest = true
        cudnn.benchmark = true

        if opt.nGPU == 1 then
            cutorch.setDevice(1)
        else
            net = utils.makeDataParallelTable(net, opt.nGPU)
        end

        criterion = criterion:cuda()
    end

    print(net)

    parameters, gradParameters = net:getParameters()
    confusion = optim.ConfusionMatrix(opt.nClass)

    -- data loader
    threads = require 'threads'
    horses = threads.Threads(opt.nhorse or 2, -- horse is faster than donkey!
        function()
            require 'torch'
        end,
        function(idx)
            print('init thread '..idx)
            dofile('listdataset.lua')
        end
    )
end

----------------------------------------------------------------
-- training
--
function xtorch.train(opt)
    if opt.backend=='GPU' then cutorch.synchronize() end
    net:training()

    -- parse arguments
    local nEpoch = opt.nEpoch
    local batchSize = opt.batchSize
    local optimState = opt.optimState
    local dataset = opt.dataset
    local c = require 'trepl.colorize'

    -- epoch tracker
    epoch = (epoch or 0) + 1
    print(string.format(c.Cyan 'Epoch %d/%d', epoch, nEpoch))

    -- do one epoch
    trainLoss = 0
    local epochSize = math.floor(dataset.ntrain/opt.batchSize)
    local bs = opt.batchSize
    for i = 1,epochSize do
        horses:addjob(
            -- the job callback (runs in data-worker thread)
            function()
                local inputs, targets = dataset:sample(bs)
                return inputs, targets
            end,
            -- the end callback (runs in the main thread)
            function (inputs, targets)
                -- synchronize for each batch training
                if opt.backend=='GPU' then cutorch.synchronize() end
                -- if use GPU, convert to cuda tensor
                targets = opt.backend=='GPU' and targets:cuda() or targets

                feval = function(x)
                    if x~= parameters then
                        parameters:copy(x)
                    end
                    gradParameters:zero()

                    local outputs = net:forward(inputs)
                    local f = criterion:forward(outputs, targets)
                    local df_do = criterion:backward(outputs, targets)
                    net:backward(inputs, df_do)
                    trainLoss = trainLoss + f

                    -- display progress & loss
                    confusion:batchAdd(outputs, targets)
                    confusion:updateValids()
                    utils.progress(i, epochSize, trainLoss/i, confusion.totalValid)
                    return f, gradParameters
                end
                optim.sgd(feval, parameters, optimState)
                if opt.backend=='GPU' then cutorch.synchronize() end
            end
        )
    end

    if opt.verbose then print(confusion) end
    confusion:zero()     -- reset confusion for test
    horses:synchronize() -- wait all horses back
    if opt.backend=='GPU' then cutorch.synchronize() end
end

----------------------------------------------------------------
-- test
--
function xtorch.test(opt)
    if opt.backend=='GPU' then cutorch.synchronize() end
    net:evaluate()

    local dataset = opt.dataset
    local epochSize = math.floor(dataset.ntest/opt.batchSize)
    local bs = opt.batchSize

    testLoss = 0
    for i = 1, epochSize do
        horses:addjob(
            function()
                local inputs, targets = dataset:get(bs*(i-1)+1, bs*i)
                return inputs, targets
            end,
            function(inputs, targets)
                if opt.backend=='GPU' then cutorch.synchronize() end
                targets = opt.backend=='GPU' and targets:cuda() or targets

                -- cutorch.synchronize()
                local outputs = net:forward(inputs)
                local f = criterion:forward(outputs, targets)
                -- cutorch.synchronize()
                testLoss = testLoss + f

                -- display progress
                confusion:batchAdd(outputs, targets)
                confusion:updateValids()
                utils.progress(i, epochSize, testLoss/i, confusion.totalValid)
                if opt.backend=='GPU' then cutorch.synchronize() end
            end
        )
    end

    if opt.verbose then print(confusion) end
    confusion:zero()
    horses:synchronize()
    if opt.backend=='GPU' then cutorch.synchronize() end
    print('\n')
end

return xtorch
