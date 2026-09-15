# Deterministic Computation

## Introduction

During application development, the results of multiple runs of the same operator may not be completely consistent due to differences such as CANN versions or NPU models. Even under the same conditions with the same random seed, the results of multiple NPU executions may differ. This difference usually originates from asynchronous multi-threaded execution in the operator implementation, which causes the floating-point accumulation order to change.

In certain special scenarios (such as experiments, debugging, and regression testing), operators need to produce identical outputs across multiple runs to facilitate problem locating or algorithm analysis. This solution is commonly referred to as "deterministic computation".

## Notes

- It is generally recommended NOT to enable deterministic computation, because deterministic computation of the same operator is usually slower than non-deterministic computation, which may degrade the single-run performance of the model. However, in scenarios such as experiments, debugging, and regression testing, where consistent results across multiple runs are required to locate problems or analyze algorithms, deterministic computation can improve efficiency.

- Thread description: The deterministic state can be set only once in the same thread. If it is set multiple times, the last effective setting takes precedence. An effective setting means that after the deterministic state is set, an operator task is actually dispatched. If the state is set but no operator task is dispatched, the deterministic variable is enabled but not delivered to the operator, and the operator is not executed. Setting determinism multiple times in one thread is not recommended for now. This issue exists regardless of whether the binary is enabled or disabled, and will be resolved in a future version.

## Usage

Currently, the mainstream invocation methods for CANN operators are aclnn APIs and PyTorch APIs (torch_extension). Some operator APIs are deterministic by default, while others are non-deterministic by default. For operators with non-deterministic implementations, some can be manually configured to enable deterministic computation.

- **Invoking aclnn APIs**

  In this scenario, enable deterministic computation through the "Runtime Configuration > aclrtSetSysParamOpt" interface (process-level) in the [Runtime API](https://hiascend.com/document/redirect/CannCommunityRuntimeApi). Specifically, set `ACL_OPT_DETERMINISTIC=1` to enable deterministic computation.

- **Invoking PyTorch APIs**

  In this scenario, enable deterministic computation through the native PyTorch switch (`torch.use_deterministic_algorithms`).

For operator APIs of different frameworks, refer to the specific aclnn API or PyTorch API documentation for their default deterministic computation policies.
