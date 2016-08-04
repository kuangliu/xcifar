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
    parameters, gradParameters = net:getParameters()
    criterion = nn.CrossEntropyCriterion()
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
    epochSize = 10
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
                -- local inputs = X_batch:float()
                -- local targets = Y_batch:cuda()
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
            end
        )
    end

    if opt.verbose then print(confusion) end
    confusion:zero()     -- reset confusion for test
    horses:synchronize() -- wait all horses back
end

----------------------------------------------------------------
-- test
--
function xtorch.test(opt)
    net:evaluate()

    local dataset = opt.dataset
    local epochSize = math.floor(dataset.ntest/opt.batchSize)
    epochSize = 10
    local bs = opt.batchSize

    testLoss = 0
    for i = 1, epochSize do
        horses:addjob(
            function()
                local inputs, targets = dataset:get(bs*(i-1)+1, bs*i)
                return inputs, targets
            end,
            function(inputs, targets)
                -- cutorch.synchronize()
                local outputs = net:forward(inputs)
                local f = criterion:forward(outputs, targets)
                -- cutorch.synchronize()
                testLoss = testLoss + f

                -- display progress
                confusion:batchAdd(outputs, targets)
                confusion:updateValids()
                utils.progress(i, epochSize, testLoss/i, confusion.totalValid)
            end
        )
    end

    if opt.verbose then print(confusion) end
    confusion:zero()
    horses:synchronize()
    print('\n')
end

return xtorch
