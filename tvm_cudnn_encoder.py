import torch
import torch.nn as nn
import tvm
from tvm import relay
from tvm.relay.backend import graph_executor_codegen

import time
import numpy as np

from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor


target = "cuda -libs=cudnn"
# target = "cuda"

embed_dim = 256
num_heads = 8

src = torch.randn(1050, 1, 256).cuda()
tgt = torch.randn(100, 1, 256).cuda()
query = torch.randn(1050, 1, 256).cuda()
key = torch.randn(1050, 1, 256).cuda()
value = torch.randn(1050, 1, 256).cuda()

encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8).cuda()
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).cuda()
model3 = transformer_encoder.eval()

scripted_model3 = torch.jit.trace(model3.forward, src).eval()



input_name0 = "0"
# input_name1 = "1"
shape_list = [(input_name0, src.shape)]

mod, params = relay.frontend.from_pytorch(scripted_model3, shape_list)

print(mod)


lib = relay.build_module.build(mod, target, params=params)
dev = tvm.device(target, 0)


dtype = "float32"

module = graph_executor.GraphModule(lib["default"](dev))


# src = src.numpy()
# tgt = tgt.numpy()

src_tvm = tvm.nd.array((np.random.random(size=src.shape)).astype(dtype), device=dev)
# tgt_tvm = tvm.nd.array((np.random.random(size=tgt.shape)).astype(dtype), device=dev)


# src_input = tvm.nd.array(src.astype(dtype), device=dev)
# tgt_input = tvm.nd.array(tgt.astype(dtype), device=dev)

# print(src_input, tgt_input)

# input_dict = {input_name0: src_tvm, input_name1: tgt_tvm}
input_dict = {input_name0: src_tvm}

# input_dict = {input_name0, src_input, input_name1, tgt_input}


module.set_input(**input_dict)

time_start = time.time()
for i in range(100):
    module.run()
print("time cost of cudnn:", (time.time() - time_start)*1000/100,"ms")

module.run()

# print()


# print(mod)

import pdb;pdb.set_trace()

target = tvm.target.Target("cuda")
dev = tvm.gpu()

# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target=target, params=params)



# opt_mode, _ = relay.optimize(mod, target, params)
# grc = graph_executor_codegen.GraphExecutorCodegen(None, target)
# grc.codegen(opt_mode["main"])
# graph_json, _, _ = grc.codegen(opt_mode["main"])
# with open("/home/yangbai/project/TBS/opt_mod_fused_softmax_batch_matmul_dense.json", "w") as f:
#    f.write(graph_json)


# import pdb; pdb.set_trace()

# tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, include_simple_tasks=True)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, include_simple_tasks=False)
print(task_weights)
# task_weights = [6, 6, 6, 6, 6, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 12, 12]

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)
    # print(task.compute_dag.flop_ct)

# import pdb;pdb.set_trace()

network = "encoder"
batch_size = 1
layout = "NHWC"
# layout = "NCHW"
target = tvm.target.Target("cuda")
dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)




def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=10000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

run_tuning()



print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# Create graph executor
# dev = tvm.device(str(target), 0)
# module = graph_executor.GraphModule(lib["default"](dev))
# data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
# module.set_input("data", data_tvm)


dtype = "float32"

module = graph_executor.GraphModule(lib["default"](dev))


# src = src.numpy()
# tgt = tgt.numpy()

src_tvm = tvm.nd.array((np.random.random(size=src.shape)).astype(dtype), device=dev)
# tgt_tvm = tvm.nd.array((np.random.random(size=tgt.shape)).astype(dtype), device=dev)


# src_input = tvm.nd.array(src.astype(dtype), device=dev)
# tgt_input = tvm.nd.array(tgt.astype(dtype), device=dev)

# print(src_input, tgt_input)

# input_dict = {input_name0: src_tvm, input_name1: tgt_tvm}
input_dict = {input_name0: src_tvm}

# input_dict = {input_name0, src_input, input_name1, tgt_input}


module.set_input(**input_dict)



# Evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=500)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
    
