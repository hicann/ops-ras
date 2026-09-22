# aclnnMatmulAbftVerify

## Product Support

<!-- npu="950" id1 -->
- <term>Ascend 950PR and 950DT series products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 series products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 series products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function Description

- API function: implements a GEMM fault-tolerant detection operator based on the variance estimation adaptive gate algorithm (V-ABFT). The operator receives matrices A and B and the pre-computed matrix multiplication result C, performs block ABFT verification on C, detects silent computation errors, and outputs per-row detection results.

- Features

  - Adaptive threshold: The operator automatically determines the threshold for comparison based on the matrix size and value range, ensuring the detection rate while avoiding false positives.
    - For the adaptive threshold algorithm and derivation, refer to the documentation at https://gitee.com/yihenggao/v-abft

  - The computation volume is significantly less than that of recomputation-based fault-tolerant solutions. When the matrix dimensions m = n = k = a, this operator requires only 8a^2 floating-point computations, while recomputation requires 2a^3 floating-point computations. The fault-tolerant threshold is also matched to this algorithm.


- Computation formula:

  $$
    C = A \times B, \quad A \in \mathbb{R}^{M \times K},\; B \in \mathbb{R}^{K \times N}
  $$

  $$
    C^r = C\times r, B^r=B\times r
  $$


  The threshold $Threshold_i$ is dynamically estimated from the local statistical features (mean and standard deviation) of the input matrices A and B, without relying on the output of matrix C.

- Operator function description:

  - The input matrices A, B, and the pre-computed C go through the checksum encoding, threshold estimation, and verification comparison process. The output is the compressed per-row fault detection result tensor `comp_row`, where 1 indicates correct and 0 indicates an error is detected.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMatmulAbftVerifyGetWorkspaceSize" API to obtain the input parameters and compute the required workspace size based on the computation flow, and then call the "aclnnMatmulAbftVerify" API to execute the computation.

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

- **Parameter Description:**

  <table style="undefined;table-layout: fixed;width: 1540px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 118px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th><a href="../../../docs/en/context/数据格式.md" target="_blank">Data Format</a></th>
        <th style="white-space: nowrap">Shape</th>
        <th><a href="../../../docs/en/context/非连续的Tensor.md" target="_blank">Non-contiguous Tensor</a></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>a (aclTensor)</td>
        <td>Input</td>
        <td>Matrix multiplication input A.</td>
        <td>
          <ul>
            <li>The dimension is 2, and the shape is [M, K].</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[M, K]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>b (aclTensor)</td>
        <td>Input</td>
        <td>Matrix multiplication input B.</td>
        <td>
          <ul>
            <li>The dimension is 2, and the shape is [K, N].</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[K, N]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>c (aclTensor)</td>
        <td>Input</td>
        <td>Pre-computed matrix multiplication result C = A x B, used as the data input for fault-tolerant detection.</td>
        <td>
          <ul>
            <li>The dimension is 2, and the shape is [M, N].</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[M, N]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>checksumWeight (aclTensor)</td>
        <td>Input</td>
        <td>Column encoding value vector, corresponding to the row checksum encoding vector $r$ (weighted vector) in ABFT.</td>
        <td>
          <ul>
            <li>The dimension is 1, and the shape is [N].</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[N]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eMax (double)</td>
        <td>Input</td>
        <td>Error threshold coefficient that controls the sensitivity of fault detection.</td>
        <td>
          <ul>
            <li>The default value is 0.001.</li>
            <li>When A and B are in BF16 precision, the recommended value is 0.001.</li>
            <li>When A and B are in FP32 precision, the recommended value is 0.00002.</li>
            <li>Using an eMax value smaller than the recommended value increases the detection rate but may also cause false positives.</li>
            <li>When false positives occur, increase the eMax value.</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compRow (aclTensor)</td>
        <td>Output</td>
        <td>Compressed row-direction fault detection bitstream output.</td>
        <td>
          <ul>
            <li>Each bit represents the detection result of one row segment. 1 indicates correct and 0 indicates an error is detected.</li>
          </ul>
        </td>
        <td>UINT8</td>
        <td>ND</td>
        <td>[ceil(M/8) * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t)</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be allocated on the device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor)</td>
        <td>Output</td>
        <td>Returns the operator executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  Where $splitN = \lceil N / 256\rceil$.

- **Return Value:**

  Returns the aclnnStatus status code. For details, refer to [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation. An error is reported in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A required input, output, or attribute is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types and data formats of a, b, c, weight, eMax, beSplitFactor, and compRow are not within the supported range.</td>
    </tr>
    <tr>
      <td>The dimensions of a and b are not 2.</td>
    </tr>
    <tr>
      <td>The first dimension (K) of a is not equal to the zeroth dimension (K) of b.</td>
    </tr>
    <tr>
      <td>eMax is negative.</td>
    </tr>
    <tr>
      <td>reduceCores is less than 0.</td>
    </tr>
  </tbody></table>

## aclnnMatmulAbftVerify

- **Parameter Description:**
    <table>
    <thead>
      <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>Input</td><td>Workspace memory address allocated on the device side.</td></tr>
      <tr><td>workspaceSize</td><td>Input</td><td>Workspace size allocated on the device side, obtained from the first-phase API aclnnMatmulAbftVerifyGetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td>Operator executor, which contains the operator computation flow.</td></tr>
      <tr><td>stream</td><td>Input</td><td>Stream used to execute the task.</td></tr>
    </tbody>
    </table>

- **Return Value:**

    Returns the aclnnStatus status code. For details, refer to [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Precautions

- Deterministic computation:
  - aclnnMatmulAbftVerify uses the deterministic implementation by default.
- The input matrices a, b, and c must be 2-dimensional. The shapes are [M, K], [K, N], and [M, N] respectively. The first dimension (K) of a must be equal to the zeroth dimension (K) of b.
- The shape of the input vector weight must be [N].
- Supported data type combinations:

  | a       | b       | c       | checksumWeight      |
  |:-------:|:-------:|:-------:|:-------:|
  | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT16 |
  | BFLOAT16| BFLOAT16| FLOAT32 | BFLOAT16|
  | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |

- Unsupported scenarios:
  - Data formats other than ND are not supported.
  - Non-contiguous tensors are not supported.
  - Scenarios where the dimensions of a, b, and c are not 2 are not supported.

## Workspace Usage Design

The workspace consists of two parts:

1. A fixed 16 MiB system workspace.
2. A user workspace, which contains 13 internal intermediate tensors and the FT internal temporary area placed sequentially.

During the operator tiling phase, the complete size is computed based on M, N, K, and the input precision, and returned through `workspaceSize`. The caller must allocate no less than this amount of contiguous device memory, and cannot allocate memory based only on the `compRow` size. The start address of all intermediate tensors is aligned to 32 bytes upward.

Definitions:

```text
splitN      = ceil(N / 256)
rowSplitLen = M * splitN
bStatLen    = ceil(splitN / 8) * 8 + 8
beLen       = K * splitN
align32(x)  = ceil(x / 32) * 32
```

Let `W` be the start address of the user workspace. Inside the kernel, `W = AscendC::GetUserWorkspace(workspace)`. For device address computation outside the operator, the current implementation is equivalent to `W = (uint8_t *)workspace + 16 MiB`.

All offsets below are relative to `W`. Let `S0 = 0`. The start offset of each tensor is `Oi = align32(Si)`, and the end position is `Si+1 = Oi + element count x element byte size`:

| Order | Intermediate Result | Start Address | Element Type | Element Count |
|:--:|:--|:--|:--|--:|
| 0 | z_row | `W + O0` | FLOAT32 | `rowSplitLen` |
| 1 | d_row | `W + O1` | FLOAT32 | `rowSplitLen` |
| 2 | threshold | `W + O2` | FLOAT32 | `rowSplitLen` |
| 3 | b_mean_abs | `W + O3` | FLOAT32 | `bStatLen` |
| 4 | b_mean_square | `W + O4` | FLOAT32 | `bStatLen` |
| 5 | b_var | `W + O5` | FLOAT32 | `bStatLen` |
| 6 | be | `W + O6` | Same as a | `beLen` |
| 7 | be_for_aiv | `W + O7` | FLOAT16 when a is FLOAT16, otherwise FLOAT32 | `beLen` |
| 8 | b_max_slice | `W + O8` | FLOAT32 | `beLen` |
| 9 | b_min_slice | `W + O9` | FLOAT32 | `beLen` |
| 10 | a_max | `W + O10` | FLOAT32 | `M` |
| 11 | a_mean | `W + O11` | FLOAT32 | `M` |
| 12 | a_min | `W + O12` | FLOAT32 | `M` |
FLOAT16 and BFLOAT16 elements occupy 2 bytes, and FLOAT32 elements occupy 4 bytes. After the 13th segment, 32-byte alignment is applied again. The remaining area is the FT internal temporary area with a logical size of:

```text
M * (splitN + 1) * sizeof(float)
```

The constant factor `1/K` used for AMean computation does not occupy workspace space. During the tiling phase, this scalar is generated based on the input precision. The kernel uses it to initialize the FT L1 internal buffer during the first AMean computation. Therefore, this factor cannot be copied out as a workspace intermediate result.

To debug and copy out a specific intermediate result, perform a device-to-host copy from the corresponding `W + Oi` in the table above after the operator execution is complete and before the workspace is released or reused. The copy byte size is "element count x element byte size". Reading the workspace externally is a debugging capability and is not a stable public output interface. When the layout changes, refer to the address splitting code under the `Workspace ABI` comment in [matmul_abft_verify.cpp](../op_kernel/matmul_abft_verify.cpp), which also serves as the reference implementation for offset computation.

## Invocation Sample

The following uses FLOAT32 input as an example. The sample first calls `aclnnGemm` to compute `C = A x B`, and then passes the matrix multiplication result `C` to `aclnnMatmulAbftVerify` for verification.

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_gemm.h"
#include "aclnnop/aclnn_matmul_abft_verify.h"

#define CHECK_RET(cond, action) \
    do {                        \
        if (!(cond)) {          \
            action;             \
        }                       \
    } while (0)

int64_t GetElementCount(const std::vector<int64_t>& shape)
{
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                    const std::vector<int64_t>& shape,
                    aclDataType dataType,
                    void** deviceAddr,
                    aclTensor** tensor)
{
    const uint64_t bytes = static_cast<uint64_t>(hostData.size()) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int main()
{
    constexpr int32_t deviceId = 0;
    constexpr int64_t M = 4096;
    constexpr int64_t N = 4096;
    constexpr int64_t K = 4096;
    constexpr double eMax = 0.00002;

    const int64_t splitN = (N + 255) / 256;
    const std::vector<int64_t> aShape{M, K};
    const std::vector<int64_t> bShape{K, N};
    const std::vector<int64_t> cShape{M, N};
    const std::vector<int64_t> checksumWeightShape{N};
    const std::vector<int64_t> compRowShape{((M + 7) / 8) * splitN};

    // When checksumWeight is all 1s, it corresponds to ordinary block row checksum.
    std::vector<float> hostA(GetElementCount(aShape), 0.5F);
    std::vector<float> hostB(GetElementCount(bShape), 0.25F);
    std::vector<float> hostC(GetElementCount(cShape), 0.0F);
    std::vector<float> hostChecksumWeight(GetElementCount(checksumWeightShape), 1.0F);
    std::vector<uint8_t> hostCompRow(GetElementCount(compRowShape), 0);

    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtCreateContext(&context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void* devA = nullptr;
    void* devB = nullptr;
    void* devC = nullptr;
    void* devGemmAddend = nullptr;
    void* devChecksumWeight = nullptr;
    void* devCompRow = nullptr;
    aclTensor* tensorA = nullptr;
    aclTensor* tensorB = nullptr;
    aclTensor* tensorC = nullptr;
    aclTensor* tensorGemmAddend = nullptr;
    aclTensor* tensorChecksumWeight = nullptr;
    aclTensor* tensorCompRow = nullptr;

    ret = CreateAclTensor(hostA, aShape, ACL_FLOAT, &devA, &tensorA);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostB, bShape, ACL_FLOAT, &devB, &tensorB);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostC, cShape, ACL_FLOAT, &devC, &tensorC);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostC, cShape, ACL_FLOAT,
                          &devGemmAddend, &tensorGemmAddend);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostChecksumWeight, checksumWeightShape, ACL_FLOAT,
                          &devChecksumWeight, &tensorChecksumWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostCompRow, compRowShape, ACL_UINT8,
                          &devCompRow, &tensorCompRow);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Preceding matrix multiplication: C = 1.0 * A * B + 0.0 * gemmAddend.
    uint64_t gemmWorkspaceSize = 0;
    aclOpExecutor* gemmExecutor = nullptr;
    ret = aclnnGemmGetWorkspaceSize(
        tensorA, tensorB, tensorGemmAddend,
        1.0F, 0.0F, 0, 0, tensorC, 0,
        &gemmWorkspaceSize, &gemmExecutor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void* gemmWorkspace = nullptr;
    if (gemmWorkspaceSize > 0) {
        ret = aclrtMalloc(&gemmWorkspace, gemmWorkspaceSize,
                          ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnGemm(gemmWorkspace, gemmWorkspaceSize, gemmExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Perform ABFT verification on the preceding GEMM result.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMatmulAbftVerifyGetWorkspaceSize(
        tensorA, tensorB, tensorC, tensorChecksumWeight,
        eMax, tensorCompRow, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize,
                          ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnMatmulAbftVerify(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclDestroyTensor(tensorA);
    aclDestroyTensor(tensorB);
    aclDestroyTensor(tensorC);
    aclDestroyTensor(tensorGemmAddend);
    aclDestroyTensor(tensorChecksumWeight);
    aclDestroyTensor(tensorCompRow);
    aclrtFree(devA);
    aclrtFree(devB);
    aclrtFree(devC);
    aclrtFree(devGemmAddend);
    aclrtFree(devChecksumWeight);
    aclrtFree(devCompRow);
    if (gemmWorkspace != nullptr) {
        aclrtFree(gemmWorkspace);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}

```
