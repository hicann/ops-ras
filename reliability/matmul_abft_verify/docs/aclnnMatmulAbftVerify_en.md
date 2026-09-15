# aclnnMatmulAbftVerify

## Product Support

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 Training Series/Atlas A3 Inference Series</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 Training Series/Atlas A2 Inference Series</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 Inference Products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas Inference Series</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas Training Series</term>: Not supported
<!-- end id6 -->

## Overview

- Function: Implements a GEMM fault-detection operator based on the variance-estimation adaptive-threshold algorithm (V-ABFT). The operator accepts matrices A and B and the precomputed matrix multiplication result C, performs block-wise ABFT verification on C, detects silent data corruption, and outputs a per-row detection result.

- Features:

  - Adaptive threshold: The operator automatically determines the comparison threshold based on the matrix dimensions and value range. This maintains the detection rate while avoiding false positives. For details about the adaptive-threshold algorithm and its derivation, see the [V-ABFT documentation](https://gitee.com/yihenggao/v-abft).
  - Significantly less computation than fault-tolerance schemes based on recomputation. When `M = N = K = a`, this operator requires only `8a^2` floating-point operations, whereas recomputation requires `2a^3` floating-point operations. The fault-detection threshold is also designed for this algorithm.

- Formulas:

  $$
    C = A \times B, \quad A \in \mathbb{R}^{M \times K},\; B \in \mathbb{R}^{K \times N}
  $$

  $$
    C^r = C \times r, \quad B^r = B \times r
  $$

  The threshold $Threshold_i$ is dynamically estimated from local statistics (mean and standard deviation) of input matrices A and B. It does not depend on the output values of matrix C.

- Operator behavior:

  - Matrices A and B and the precomputed matrix C pass through checksum encoding, threshold estimation, and verification. The operator outputs the compressed per-row fault-detection tensor `compRow`. A bit value of 1 indicates a correct result, and 0 indicates that an error has been detected.

## Function Prototypes

This operator uses a [two-stage interface](../../../docs/en/context/two_phase_api.md). First call `aclnnMatmulAbftVerifyGetWorkspaceSize` to validate the inputs and calculate the required workspace size, and then call `aclnnMatmulAbftVerify` to execute the operator.

```cpp
aclnnStatus aclnnMatmulAbftVerifyGetWorkspaceSize(
    const aclTensor  *a,
    const aclTensor  *b,
    const aclTensor  *c,
    const aclTensor  *checksumWeight,
    double            eMax,
    const aclTensor  *compRow,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor);
```

```cpp
aclnnStatus aclnnMatmulAbftVerify(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream     stream);
```

## aclnnMatmulAbftVerifyGetWorkspaceSize

- **Parameters:**

  | Parameter | Input/Output | Description | Usage | Data Type | Format | Shape | Non-contiguous Tensor |
  |:--|:--:|:--|:--|:--|:--:|:--|:--:|
  | `a` (`aclTensor`) | Input | Input matrix A. | Must be a 2D tensor with shape `[M, K]`. | FLOAT16, BFLOAT16, FLOAT32 | ND | `[M, K]` | No |
  | `b` (`aclTensor`) | Input | Input matrix B. | Must be a 2D tensor with shape `[K, N]`. | FLOAT16, BFLOAT16, FLOAT32 | ND | `[K, N]` | No |
  | `c` (`aclTensor`) | Input | Precomputed matrix multiplication result `C = A × B`, used as the data to be verified. | Must be a 2D tensor with shape `[M, N]`. | FLOAT32 | ND | `[M, N]` | No |
  | `checksumWeight` (`aclTensor`) | Input | Column-encoding vector corresponding to the row-checksum encoding vector $r$ (weight vector) in ABFT. | Must be a 1D tensor with shape `[N]`. | FLOAT16, BFLOAT16, FLOAT32 | ND | `[N]` | No |
  | `eMax` (`double`) | Input | Error-threshold coefficient that controls fault-detection sensitivity. | The default value is `0.001`. The recommended value is `0.001` for BF16 inputs and `0.00002` for FP32 inputs. A smaller value increases the detection rate but may cause false positives. Increase this value if false positives occur. | - | - | - | - |
  | `compRow` (`aclTensor`) | Output | Compressed row-direction fault-detection bitstream. | Each bit represents the detection result of one row segment. A value of 1 indicates a correct result, and 0 indicates an error. | UINT8 | ND | `[ceil(M/8) * splitN]` | No |
  | `workspaceSize` (`uint64_t`) | Output | Size of the workspace that must be allocated on the Device. | - | - | - | - | - |
  | `executor` (`aclOpExecutor`) | Output | Operator executor containing the operator execution process. | - | - | - | - | - |

  Here, $splitN = \lceil N / 256 \rceil$.

- **Return Value:**

  Returns an `aclnnStatus` status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md). The first-stage interface validates the input arguments and reports an error in the following cases:

  | Return Value | Error Code | Description |
  |:--|:--:|:--|
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | A required input, output, or attribute is a null pointer. |
  | ACLNN_ERR_PARAM_INVALID | 161002 | The data type or format of `a`, `b`, `c`, `weight`, `eMax`, `beSplitFactor`, or `compRow` is unsupported. |
  | ACLNN_ERR_PARAM_INVALID | 161002 | `a` or `b` is not a 2D tensor. |
  | ACLNN_ERR_PARAM_INVALID | 161002 | Dimension 1 (K) of `a` differs from dimension 0 (K) of `b`. |
  | ACLNN_ERR_PARAM_INVALID | 161002 | `eMax` is negative. |
  | ACLNN_ERR_PARAM_INVALID | 161002 | `reduceCores` is less than 0. |

## aclnnMatmulAbftVerify

- **Parameters:**

  | Parameter | Input/Output | Description |
  |:--|:--:|:--|
  | `workspace` | Input | Address of the workspace allocated on the Device. |
  | `workspaceSize` | Input | Size of the workspace allocated on the Device, obtained from `aclnnMatmulAbftVerifyGetWorkspaceSize`. |
  | `executor` | Input | Operator executor containing the operator execution process. |
  | `stream` | Input | Stream on which the task is executed. |

- **Return Value:**

  Returns an `aclnnStatus` status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: `aclnnMatmulAbftVerify` uses a deterministic implementation by default.
- Input matrices `a`, `b`, and `c` must be 2D tensors with shapes `[M, K]`, `[K, N]`, and `[M, N]`, respectively. Dimension 1 (K) of `a` must equal dimension 0 (K) of `b`.
- The shape of input vector `checksumWeight` must be `[N]`.
- The following data type combinations are supported:

  | a | b | c | checksumWeight |
  |:--:|:--:|:--:|:--:|
  | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT16 |
  | BFLOAT16 | BFLOAT16 | FLOAT32 | BFLOAT16 |
  | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |

- Unsupported scenarios:

  - Formats other than ND are not supported.
  - Non-contiguous tensors are not supported.
  - Non-2D `a`, `b`, or `c` tensors are not supported.

## Workspace Design

The workspace consists of two parts:

1. A fixed 16 MiB system workspace.
2. A user workspace containing 13 internal intermediate tensors followed by the FT internal temporary area.

During tiling, the operator calculates the complete size from M, N, K, and the input precision and returns it through `workspaceSize`. The caller must allocate at least this amount of contiguous Device memory. Allocating only enough memory for `compRow` is insufficient. The starting address of every intermediate tensor is aligned upwards to 32 bytes.

Definitions:

```text
splitN      = ceil(N / 256)
rowSplitLen = M * splitN
bStatLen    = ceil(splitN / 8) * 8 + 8
beLen       = K * splitN
align32(x)  = ceil(x / 32) * 32
```

Let `W` be the start address of the user workspace. In the kernel, `W = AscendC::GetUserWorkspace(workspace)`. For Device-address calculations outside the operator, the current implementation is equivalent to `W = (uint8_t *)workspace + 16 MiB`.

All offsets below are relative to `W`. Let `S0 = 0`. The starting offset of each tensor is `Oi = align32(Si)`, and its end position is `Si+1 = Oi + element count × element size`:

| Order | Intermediate Result | Start Address | Element Type | Element Count |
|:--:|:--|:--|:--|--:|
| 0 | z_row | `W + O0` | FLOAT32 | `rowSplitLen` |
| 1 | d_row | `W + O1` | FLOAT32 | `rowSplitLen` |
| 2 | threshold | `W + O2` | FLOAT32 | `rowSplitLen` |
| 3 | b_mean_abs | `W + O3` | FLOAT32 | `bStatLen` |
| 4 | b_mean_square | `W + O4` | FLOAT32 | `bStatLen` |
| 5 | b_var | `W + O5` | FLOAT32 | `bStatLen` |
| 6 | be | `W + O6` | Same as `a` | `beLen` |
| 7 | be_for_aiv | `W + O7` | FLOAT16 when `a` is FLOAT16; otherwise FLOAT32 | `beLen` |
| 8 | b_max_slice | `W + O8` | FLOAT32 | `beLen` |
| 9 | b_min_slice | `W + O9` | FLOAT32 | `beLen` |
| 10 | a_max | `W + O10` | FLOAT32 | `M` |
| 11 | a_mean | `W + O11` | FLOAT32 | `M` |
| 12 | a_min | `W + O12` | FLOAT32 | `M` |

FLOAT16 and BFLOAT16 elements occupy 2 bytes, and FLOAT32 elements occupy 4 bytes. After the 13th segment, the address is aligned upwards to 32 bytes again. The remaining region is the FT internal temporary area, whose logical size is:

```text
M * (splitN + 1) * sizeof(float)
```

The constant factor `1/K` used to calculate AMean does not occupy workspace. During tiling, the scalar is generated according to the input precision. When AMean is first calculated, the kernel uses the scalar directly to initialize the FT L1 internal buffer. Therefore, this factor cannot be copied out as a workspace intermediate result.

To debug and copy an intermediate result, perform a Device-to-Host copy starting at the corresponding `W + Oi` address in the table after the operator has finished and before the workspace is released or reused. The number of bytes to copy is `element count × element size`. Reading the workspace externally is a debugging capability and is not a stable public output interface. If the layout changes, refer to the address-partitioning code below the `Workspace ABI` comment in [matmul_abft_verify.cpp](../op_kernel/matmul_abft_verify.cpp). That code is also an example of the offset calculation.

## Example

The following example uses FLOAT32 inputs. It first calls `aclnnGemm` to calculate `C = A × B`, and then passes C to `aclnnMatmulAbftVerify` for verification. A complete runnable example is provided in [test_aclnn_matmul_abft_verify_bf16.cpp](../examples/test_aclnn_matmul_abft_verify_bf16.cpp).
