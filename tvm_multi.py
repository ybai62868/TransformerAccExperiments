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

embed_dim = 256
num_heads = 8

src = torch.randn(1050, 1, 256).cuda()
tgt = torch.randn(100, 1, 256).cuda()
query = torch.randn(1050, 1, 256).cuda()
key = torch.randn(1050, 1, 256).cuda()
value = torch.randn(1050, 1, 256).cuda()

multihead_attn = nn.MultiheadAttention(embed_dim, num_heads).cuda()
    # attn_output, attn_output_weights = multihead_attn(query, key, value)
    # print(attn_output.shape, attn_output_weights.shape)
model = multihead_attn.eval()

scripted_model = torch.jit.trace(model.forward, (query, key, value)).eval()



input_name0 = "0"
input_name1 = "1"
input_name2 = "2"
shape_list = [(input_name0, src.shape), (input_name1, key.shape), (input_name2, value.shape)]

mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# print(mod)

# import pdb;pdb.set_trace()

target = tvm.target.Target("cuda")
dev = tvm.gpu()

# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target=target, params=params)



# opt_mode, _ = relay.optimize(mod, target, params)
# grc = graph_executor_codegen.GraphExecutorCodegen(None, target)
# grc.codegen(opt_mode["main"])
# graph_json, _, _ = grc.codegen(opt_mode["main"])
# with open("/home/yangbai/project/TBS/opt_mod_fused.json", "w") as f:
#    f.write(graph_json)


# import pdb; pdb.set_trace()

tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, include_simple_tasks=True)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

# import pdb;pdb.set_trace()

network = "multihead"
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
        num_measure_trials=200,  # change this to 20000 to achieve the best performance
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

q_tvm = tvm.nd.array((np.random.random(size=query.shape)).astype(dtype), device=dev)
k_tvm = tvm.nd.array((np.random.random(size=key.shape)).astype(dtype), device=dev)
v_tvm = tvm.nd.array((np.random.random(size=value.shape)).astype(dtype), device=dev)
# tgt_tvm = tvm.nd.array((np.random.random(size=tgt.shape)).astype(dtype), device=dev)


# src_input = tvm.nd.array(src.astype(dtype), device=dev)
# tgt_input = tvm.nd.array(tgt.astype(dtype), device=dev)

# print(src_input, tgt_input)

# input_dict = {input_name0: src_tvm, input_name1: tgt_tvm}
input_dict = {input_name0: q_tvm, input_name1: k_tvm, input_name2: v_tvm}

# input_dict = {input_name0, src_input, input_name1, tgt_input}


module.set_input(**input_dict)



# Evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=500)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
    
 
 # multi-head attention
