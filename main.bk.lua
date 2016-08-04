require 'pl'
require 'nn'
require 'cunn'
require 'xlua'
require 'image'
require 'optim'
require 'cutorch'

dofile('vgg.lua')
dofile('checkpoint.lua')
dofile('xlog.lua')
dofile('resnet.lua')
dofile('horse.lua') -- load trainLoader, testLoader

cudnn = require 'cudnn'
threads = require 'threads'
c = require 'trepl.colorize'

opt = lapp[[
    -n,--nGPU              (default 1)                   num of GPUs to use
    -c,--checkpointPath    (default './checkpoint/')     checkpoint saving path
    -b,--batchSize         (default 128)                 batch size
    -r,--resume                                          resume from checkpoint
    ]]

horses = threads.Threads(2,
    function()
        require 'torch'
    end,
    function(idx)
        print('init thread '..idx)
        dofile('horse.lua')
    end
)

function initModel()
	--local resnet = getResNet30()
	--local resnet = getVGG()
	local resnet = torch.load('./model/xxxx.t7')

	-- this replaces 'nn'modules with 'cudnn' counterparts in-place
	cudnn.convert(resnet, cudnn):cuda()

	-- cudnn will optimize self for effciency
	cudnn.fastest = true
	cudnn.benchmark = true

	-- init whole net
	local net = nn.Sequential()
			:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'))

	if opt.nGPU == 1 then
	    -- use single GPU, use the first one in default
	    net:add(resnet)
	    cutorch.setDevice(1)  -- change the GPU ID as you like
	else
	    -- multi-GPU, use GPU #1,#2,...#n
		local gpus = torch.range(1, opt.nGPU):totable()

		local dpt = nn.DataParallelTable(1,true,true)
					  :add(resnet,gpus)
					  :threads(function()
					      local cudnn = require 'cudnn'
						  cudnn.fastest, cudnn.benchmark = true, true
					  end)
		net:add(dpt:cuda())
	end

	return net
end


-- Set up model
print(c.blue '==> '..'setting up model..')

if opt.resume then
	print('resuming...')
	latest = checkpoint.load()
	epoch = latest.epoch
	net = torch.load(latest.modelFile)
    optimState = torch.load(latest.optimFile)
	optimState.learningRate = 0.0001
    bestTestLoss = latest.bestTestLoss
else
	net = initModel()
end

print(net)

parameters, gradParameters = net:getParameters()
--criterion = nn.SmoothL1Criterion():cuda()
criterion = nn.MSECriterion():cuda()

-- Load data
--print(c.blue '==> '..'loading data..')

testLogger = xLogger(paths.concat('log', 'test.log'))
testLogger:setNames{'Train MSE', 'Test MSE'}

-- Set up optimizer
print(c.blue '==> '..'configure optimizer..\n')
optimState = optimState or {
    learningRate = 0.001,
    learningRateDecay = 1e-7,
    weightDecay = 1e-4,
    momentum = 0.9,
    nesterov = true,
    dampening = 0.0
}

function trainBatch(inputs, targets)
    cutorch.synchronize()
    collectgarbage()

    inputs = inputs:float()
    targets = targets:cuda()

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
        print(trainLoss/ii)

        return f, gradParameters
    end
    optim.sgd(feval, parameters, optimState)
    cutorch.synchronize()
end

function train()
	print('training... lr='..optimState.learningRate)
    cutorch.synchronize()
	net:training()
    epoch = (epoch or 0)+1

    trainLoss = 0
    local epochSize = math.floor(trainLoader.nSamples/opt.batchSize)
	local sz = opt.batchSize
    for i = 1,epochSize do
		ii = i
		xlua.progress(i,epochSize)
        horses:addjob(
            -- the job callback (runs in data-worker thread)
            function()
                local inputs, targets = trainLoader:sample(sz)
                --inputs = torch.randn(opt.batchSize,3,150,150)
        		--targets = torch.randn(opt.batchSize, 10):cuda()
                return inputs, targets
            end,
            -- the end callback (runs in the main thread)
            trainBatch
        )
        -- print('data loading/training time: ' .. t1.real..'/'..(t2.real-t1.real) .. ' seconds')
    end
    horses:synchronize()
    cutorch.synchronize()
	trainLoss = trainLoss/epochSize
    --print(c.Green '==> '..loss/#indices)
end

function test()
	print('testing...')
    cutorch.synchronize()
    net:evaluate()

    testLoss = 0
    local epochSize = math.floor(testLoader.nSamples/opt.batchSize)
	local sz = opt.batchSize
    for i = 1, epochSize do
		xlua.progress(i,epochSize)
        horses:addjob(
            function()
                local inputs, targets = testLoader:get(sz*(i-1)+1, sz*i)
                return inputs, targets
            end,
            function(inputs, targets)
				cutorch.synchronize()
                local outputs = net:forward(inputs)
                local f = criterion:forward(outputs, targets:cuda())
                cutorch.synchronize()
				testLoss = testLoss+f
                print(testLoss/i)
            end
        )
    end
    cutorch.synchronize()
	horses:synchronize()

    testLoss = testLoss/epochSize
	bestTestLoss = bestTestLoss or math.huge
	if testLoss < bestTestLoss then
		print(c.Cyan '==> Find new world!')
		bestTestLoss = testLoss
		checkpoint.save(epoch, net, optimState, bestTestLoss)
	end

    print(c.Green '==> '..epoch..'\t'..trainLoss..'\t'..testLoss..'\t'..bestTestLoss)

	if testLogger then
        testLogger:add{trainLoss, testLoss}
    end
end

-- main loop
while true do
    train()
	test()
end
