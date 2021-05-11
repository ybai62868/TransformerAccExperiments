################################################################################
# Example 1: Writing and Scheduling Vector Addition in TE for CPU
# ---------------------------------------------------------------
#
# Let's look at an example in Python in which we will implement a TE for
# vector addition, followed by a schedule targeted towards a CPU. We begin by initializing a TVM
# environment.

import tvm
import tvm.testing
from tvm import te
import numpy as np



run_cuda = False
if run_cuda:
    # Change this target to the correct backend for you gpu. For example: cuda (NVIDIA GPUs),
    # rocm (Radeon GPUS), OpenCL (opencl).
    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")

    # Recreate the schedule
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    print(type(C))

    s = te.create_schedule(C.op)

    bx, tx = s[C].split(C.op.axis[0], factor=64)

    xXXXXXXXx

    ################################################################################
    # Finally we must bind the iteration axis bx and tx to threads in the GPU
    # compute grid. The naive schedule is not valid for GPUs, and these are
    # specific constructs that allow us to generate code that runs on a GPU.

    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

    ######################################################################
    # Compilation
    # -----------
    # After we have finished specifying the schedule, we can compile it
    # into a TVM function. By default TVM compiles into a type-erased
    # function that can be directly called from the python side.
    #
    # In the following line, we use tvm.build to create a function.
    # The build function takes the schedule, the desired signature of the
    # function (including the inputs and outputs) as well as target language
    # we want to compile to.
    #
    # The result of compilation fadd is a GPU device function (if GPU is
    # involved) as well as a host wrapper that calls into the GPU
    # function. fadd is the generated host wrapper function, it contains
    # a reference to the generated device function internally.

    fadd = tvm.build(s, [A, B, C], target=tgt_gpu, name="myadd")

    ################################################################################
    # The compiled TVM function is exposes a concise C API that can be invoked from
    # any language.
    #
    # We provide a minimal array API in python to aid quick testing and prototyping.
    # The array API is based on the `DLPack <https://github.com/dmlc/dlpack>`_ standard.
    #
    # - We first create a GPU device.
    # - Then tvm.nd.array copies the data to the GPU.
    # - ``fadd`` runs the actual computation
    # - ``asnumpy()`` copies the GPU array back to the CPU (so we can verify correctness).
    #
    # Note that copying the data to and from the memory on the GPU is a required step.

    dev = tvm.device(tgt_gpu.kind.name, 0)

    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

    ################################################################################
    # Inspect the Generated GPU Code
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # You can inspect the generated code in TVM. The result of tvm.build is a TVM
    # Module. fadd is the host module that contains the host wrapper, it also
    # contains a device module for the CUDA (GPU) function.
    #
    # The following code fetches the device module and prints the content code.

    if (
        tgt_gpu.kind.name == "cuda"
        or tgt_gpu.kind.name == "rocm"
        or tgt_gpu.kind.name.startswith("opencl")
    ):
        dev_module = fadd.imported_modules[0]
        print("-----GPU code-----")
        print(dev_module.get_source())
    else:
        print(fadd.get_source())

################################################################################
# Saving and Loading Compiled Modules
# -----------------------------------
# Besides runtime compilation, we can save the compiled modules into a file and
# load them back later.
#
# The following code first performs the following steps:
#
# - It saves the compiled host module into an object file.
# - Then it saves the device module into a ptx file.
# - cc.create_shared calls a compiler (gcc) to create a shared library

from tvm.contrib import cc
from tvm.contrib import utils

temp = utils.tempdir()
fadd.save(temp.relpath("myadd.o"))
if tgt.kind.name == "cuda":
    fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
if tgt.kind.name == "rocm":
    fadd.imported_modules[0].save(temp.relpath("myadd.hsaco"))
if tgt.kind.name.startswith("opencl"):
    fadd.imported_modules[0].save(temp.relpath("myadd.cl"))
cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
print(temp.listdir())

################################################################################
# .. note:: Module Storage Format
#
#   The CPU (host) module is directly saved as a shared library (.so). There
#   can be multiple customized formats of the device code. In our example, the
#   device code is stored in ptx, as well as a meta data json file. They can be
#   loaded and linked separately via import.

################################################################################
# Load Compiled Module
# ~~~~~~~~~~~~~~~~~~~~
# We can load the compiled module from the file system and run the code. The
# following code loads the host and device module separately and links them
# together. We can verify that the newly loaded function works.

fadd1 = tvm.runtime.load_module(temp.relpath("myadd.so"))
if tgt.kind.name == "cuda":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.ptx"))
    fadd1.import_module(fadd1_dev)

if tgt.kind.name == "rocm":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.hsaco"))
    fadd1.import_module(fadd1_dev)

if tgt.kind.name.startswith("opencl"):
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.cl"))
    fadd1.import_module(fadd1_dev)

fadd1(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

################################################################################
# Pack Everything into One Library
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In the above example, we store the device and host code separately. TVM also
# supports export everything as one shared library. Under the hood, we pack
# the device modules into binary blobs and link them together with the host
# code. Currently we support packing of Metal, OpenCL and CUDA modules.

fadd.export_library(temp.relpath("myadd_pack.so"))
fadd2 = tvm.runtime.load_module(temp.relpath("myadd_pack.so"))
fadd2(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

