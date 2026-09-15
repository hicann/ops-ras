# Batch Consistency

## Introduction

During application development, some operators may produce result deviations for the same Token across different batch sizes or at different positions within the same batch, in pursuit of higher performance.
Currently, for some operators, under the same runtime environment conditions, the computation process can be configured to adopt a Batch consistency algorithm, so that the computation results remain completely consistent no matter how the inputs are combined.
**Batch consistency algorithm**: For a given Token, the output must be bit-wise completely consistent, regardless of the position of the Token within the batch, the batch size, or which other Tokens are batched together with it.

## Notes

- It is generally recommended NOT to enable Batch consistency computation, because after Batch consistency computation is enabled for an operator, the relevant operators suffer a certain degree of performance degradation, and the single-run performance of the model may decrease. However, in scenarios such as experiments, debugging, and regression testing, where identical results across multiple runs are required to locate problems and experiment with algorithms, Batch consistency computation can improve efficiency.

- The current configuration is a process-level switch.

- **Version constraints**: TorchNPU version 26.2.0 or later, and CANN version 9.2.0 or later.

## Usage

Currently, the mainstream invocation methods for CANN operators are aclnn APIs and PyTorch APIs (torch_extension). Some operator APIs are Batch-consistent by default, while others are non-Batch-consistent by default. For operators with non-Batch-consistent implementations, some can be manually configured to enable Batch consistency.

- **Invoking aclnn APIs**

  In this scenario, enable Batch consistency through the "Runtime Configuration > aclrtSetSysParamOpt" interface (process-level) in the [Runtime API](https://hiascend.com/document/redirect/CannCommunityRuntimeApi). Specifically, set `ACL_OPT_DETERMINISTIC=3` to enable Batch consistency computation.

- **Invoking PyTorch APIs**

  In this scenario, enable Batch consistency computation through the `torch_npu.npu.set_deterministic_level` interface in the [TorchNPU Custom APIs](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/overview.md).

For operator APIs of different frameworks, refer to the specific aclnn API or PyTorch API documentation for their default Batch consistency policies.
