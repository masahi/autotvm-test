import os
import numpy as np
import time
from collections import namedtuple


batch_size = 1
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)
output_shape = (batch_size, 1000)
num_runs = 500
networks = ["resnet-50", "vgg-16", "densenet-121"]


def mxnet_bench(network):
    import mxnet as mx
    from mxnet.gluon.model_zoo.vision import get_model
    models = {"resnet-50" : "resnet50_v1", "vgg-16" : "vgg16", "densenet-121" :  "densenet121"}
    Batch = namedtuple('Batch', ['data'])
    block = get_model(models[network], pretrained=True)
    data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
    mx_data = mx.nd.array(data_array).as_in_context(mx.gpu())

    block.collect_params().reset_ctx(mx.gpu())
    block.hybridize()
    block(mx_data)

    if not os.path.exists("symbol"):
        os.mkdir("symbol")
    block.export("symbol/" + network)


    sym, arg_params, aux_params = mx.model.load_checkpoint("symbol/" + network, 0)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', input_shape)], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    # Warmup
    for _ in range(3):
        mod.forward(Batch([mx_data]))
        for output in mod.get_outputs():
            output.wait_to_read()

    s = time.time()
    for _ in range(num_runs):
        mod.forward(Batch([mx_data]))
        for output in mod.get_outputs():
            output.wait_to_read()

    return (time.time() - s) / num_runs / 1000


def pytorch_bench(network):
    import torch
    import torchvision.models as models
    get_workload = {"resnet-50" : models.resnet50,
                    "vgg-16" : models.vgg16,
                    "densenet-121" : models.densenet121}

    input_tensor = torch.randn(input_shape)
    model = get_workload[network](pretrained=True)
    device = "cuda"
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # warm up
    for i in range(3):
        output = model(input_tensor)
    torch.cuda.synchronize()

    import time
    t0 = time.time()

    for i in range(num_runs):
        output = model(input_tensor)

    torch.cuda.synchronize()

    return (time.time() - t0) / num_runs * 1000


def tvm_bench(network, use_cudnn=False):
    import nnvm
    import tvm
    import tvm.contrib.graph_runtime as runtime
    import nnvm.compiler
    import nnvm.testing

    get_workload = {"resnet-50" : nnvm.testing.resnet.get_workload,
                    "vgg-16" : nnvm.testing.vgg.get_workload,
                    "densenet-121" : nnvm.testing.densenet.get_workload}

    n_layer = int(network.split('-')[1])

    if use_cudnn:
        target = tvm.target.create('cuda -libs=cudnn')
    else:
        target = tvm.target.create('cuda -model=1080ti')

    net, params = get_workload[network](num_layers=n_layer, batch_size=batch_size)

    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(
            net, target=target, shape={'data': input_shape}, params=params, dtype="float32")

    ctx = tvm.context(str(target), 0)
    module = runtime.create(graph, lib, ctx)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
    module.set_input('data', data_tvm)
    module.set_input(**params)
    ftimer = module.module.time_evaluator("run", ctx, number=num_runs, repeat=1)
    return ftimer().mean * 1000


autotvm_time = []
cudnn_time = []
mxnet_time = []
torch_time = []
for network in networks:
    print("Running ", network)
    autotvm_time.append(tvm_bench(network))
    cudnn_time.append(tvm_bench(network, use_cudnn=True))
    mxnet_time.append(mxnet_bench(network))
    torch_time.append(pytorch_bench(network))

for (i, network) in enumerate(networks):
    print("Network: ", network)
    print("AutoTVM: ", autotvm_time[i])
    print("TVM + cuDNN: ", cudnn_time[i])
    print("MXNet + cuDNN: ", mxnet_time[i])
    print("Pytorch + cuDNN: %f\n" % torch_time[i])
