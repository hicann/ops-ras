# AscendOps

## Prerequisites

- Refer to [Prerequisites](../../docs/en/install/quick_install.md) to complete the basic environment setup.
- GCC 9.4.0+
- Python 3.8+
- PyTorch>=2.6.0
- Corresponding version of [TorchNPU](https://gitcode.com/Ascend/pytorch/releases)

## Installation Steps

1. Install Dependencies:
    ```sh
    python3 -m pip install -r requirements.txt
    ```

2. Build the Wheel:
    ```sh
    # -n: non-isolated build (uses existing environment)
    python3 -m build --wheel -n
    ```

3. Install Package:
    ```sh
    python3 -m pip install dist/*.whl --force-reinstall --no-deps
    ```

4. (Optional) Before rebuilding, run the following command to clean the compilation cache:
   ```sh
   python3 setup.py clean
   ```

## Quick Start
After the installation is complete, use NPU operators in the same way as ordinary PyTorch operators.

```python
import torch
import torch_npu
import ascend_ops

# Initialize data on NPU
x = torch.randn(10, 32, dtype=torch.float32).npu()
y = torch.randn(10, 32, dtype=torch.float32).npu()

# Call the custom NPU operator
npu_result = torch.ops.ascend_ops.add(x, y)

# Verify against CPU ATen implementation
cpu_x = x.cpu()
cpu_y = y.cpu()
cpu_result = cpu_x + cpu_y

assert torch.allclose(cpu_result, npu_result.cpu(), rtol=1e-6)
print("Verification successful!")
```

## Developer Guide: Adding a New Operator

To implement a new operator (for example, `add`), provide a C++ implementation.

1. Create a folder named after the operator `add` under the csrc directory. Inside this folder, create a subfolder named after the target SoC, for example, `ascend910b`.

2. Create a `CMakeLists.txt` file in the SoC directory:
    ```
    add_sources("--npu-arch=dav-2201")
    ```
    Here, `dav-2201` is the compilation parameter for the ascend910b chip.

3. Create an `add.cpp` file in the SoC directory (use the operator name as the file name). This file contains all modules required for developing an AI Core operator:
    - Operator schema registration
    - Operator meta function implementation and registration
    - Operator kernel implementation (Ascend C)
    - Operator NPU invocation implementation and registration

    ```cpp
    #include <ATen/Operators.h>
    #include <torch/all.h>
    #include <torch/library.h>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "torch_npu/csrc/framework/OpCommand.h"
    #include "kernel_operator.h"
    #include "platform/platform_ascendc.h"
    #include <type_traits>

    namespace ascend_ops {  // The current project is a namespace.
    namespace Add {         // Use an independent namespace for each operator to prevent global variable pollution.

    /**
     * Register the operator schema with the PyTorch framework.
     * The framework is aware of this operator.
     */
    // Register the operator's schema
    TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
    {
        m.def("add(Tensor x, Tensor y) -> Tensor");
    }

    /**
     * Implement the operator meta function, that is, InferShape + InferDtype.
     * Infer the output shape and required memory without actual computation.
     */
    // Meta function implementation of Add
    torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
    {
        TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
        auto z = torch::empty_like(x);
        return z;
    }

    /**
     * Register the operator meta function with the framework.
     * The framework calls this meta function to determine the required memory before the operator computation.
     * This supports torch.compile, AutoGrad, AclGraph, and other graph acceleration features.
     */
    // Register the Meta implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
    {
        m.impl("add", add_meta);
    }

    /**
     * NPU operator kernel implementation using Ascend C APIs for the target SoC.
     */
    template <typename T>
    __global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        // kernel implementation
    }

    /**
     * Implement the operator invocation interface.
     * This interface must complete the NPU kernel invocation.
     * 1. Compute the output tensor count, shape, and data type (call the meta function or implement directly).
     * 2. Compute tiling: determine the block computation based on the shape.
     * 3. Invoke the NPU kernel.
     *
     */
    torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y)
    {
        auto z = add_meta(x, y);
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        int64_t totalLength, blockDim, blockLength, tileSize;
        totalLength = x.numel();
        std::tie(blockDim, blockLength, tileSize) = calc_tiling_params(totalLength);
        auto x_ptr = (GM_ADDR)x.data_ptr();
        auto y_ptr = (GM_ADDR)y.data_ptr();
        auto z_ptr = (GM_ADDR)z.data_ptr();
        auto acl_call = [=]() -> int {
            AT_DISPATCH_SWITCH(
                x.scalar_type(), "add_npu",
                // Invoke different NPU kernels based on data types.
                AT_DISPATCH_CASE(torch::kFloat32, [&] {
                    using scalar_t = float;
                    add_kernel<scalar_t><<<blockDim, nullptr, stream>>>(x_ptr, y_ptr, z_ptr,     totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kFloat16, [&] {
                    using scalar_t = half;
                    add_kernel<scalar_t><<<blockDim, nullptr, stream>>>(x_ptr, y_ptr, z_ptr,     totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kInt32, [&] {
                    using scalar_t = int32_t;
                    add_kernel<scalar_t><<<blockDim, nullptr, stream>>>(x_ptr, y_ptr, z_ptr,     totalLength, blockLength, tileSize);
                })
            );
            return 0;
        };
        // Use the RunOpApi or RunOpApiV2 interface to ensure the invocation sequence is consistent with TorchNPU calling the aclnn interface.
        at_npu::native::OpCommand::RunOpApi("Add", acl_call);
        return z;
    }

    /**
     * Register the operator invocation function with the framework. The device is PrivateUse1.
     * The framework dispatches to this operator implementation when all inputs are on the NPU device.
     */
    // Register the NPU implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
    {
        m.impl("add", add_npu);
    }

    }  // namespace Add
    }  // namespace ascend_ops

    ```
4. Build the wheel package using the [Installation Steps](#installation-steps) section. Install and test the package.
5. For testing the operator API, refer to the implementation in [test_add.py](tests/add/test_add.py).
