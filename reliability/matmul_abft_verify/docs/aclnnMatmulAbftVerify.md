# aclnnMatmulAbftVerify

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：实现基于方差估计自适应门限算法（V-ABFT）的GEMM容错检测算子。算子接收矩阵A、B以及预先计算的矩阵乘结果C，对C进行分块ABFT校验，检测静默计算错误并输出逐行检测结果。

- 特点

  - 自适应阈值: 该算子能够根据矩阵大小与值域自动确定用于比对的阈值，能够在保证检出率的同时避免误检。
    - 自适应阈值算法与推导见https://gitee.com/yihenggao/v-abft的文档

  - 计算量显著小于基于重新计算的容错方案。在矩阵维度m=n=k=a时，本算子只需8a^2次浮点计算，而重算则需要2a^3次浮点计算。且容错阈值也与该算法相匹配


- 计算公式：

  $$
    C = A \times B, \quad A \in \mathbb{R}^{M \times K},\; B \in \mathbb{R}^{K \times N}
  $$

  $$
    C^r = C\times r, B^r=B\times r
  $$


  其中阈值$Threshold_i$由输入矩阵A、B的局部统计特征（均值、标准差）动态估计，无需依赖C矩阵输出结果。

- 算子功能说明：

  - 输入矩阵A、B和预先计算的C经过checksum编码、阈值估计和校验比对流程，输出压缩后的逐行故障检测结果张量`comp_row`，1表示正确，0表示检测到错误。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMatmulAbftVerifyGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnMatmulAbftVerify”接口执行计算。

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

- **参数说明：**

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
        <th>参数名</th>
        <th style="white-space: nowrap">输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th><a href="../../../docs/zh/context/数据格式.md" target="_blank">数据格式</a></th>
        <th style="white-space: nowrap">维度(shape)</th>
        <th><a href="../../../docs/zh/context/非连续的Tensor.md" target="_blank">非连续的Tensor</a></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>a（aclTensor）</td>
        <td>输入</td>
        <td>矩阵乘法输入A。</td>
        <td>
          <ul>
            <li>维度为2，shape为[M, K]。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[M, K]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>b（aclTensor）</td>
        <td>输入</td>
        <td>矩阵乘法输入B。</td>
        <td>
          <ul>
            <li>维度为2，shape为[K, N]。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[K, N]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>c（aclTensor）</td>
        <td>输入</td>
        <td>预先计算的矩阵乘结果C = A × B，作为容错检测的数据输入。</td>
        <td>
          <ul>
            <li>维度为2，shape为[M, N]。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[M, N]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>checksumWeight（aclTensor）</td>
        <td>输入</td>
        <td>列编码值向量，对应ABFT中的行校验和编码向量$r$（加权向量）。</td>
        <td>
          <ul>
            <li>维度为1，shape为[N]。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[N]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eMax（double）</td>
        <td>输入</td>
        <td>误差阈值系数，控制故障检测的灵敏度。</td>
        <td>
          <ul>
            <li>默认值为0.001。</li>
            <li>A,B为BF16精度,推荐值:0.001</li>
            <li>A,B为FP32精度, 推荐值: 0.00002</li>
            <li>使用小于推荐值的eMax会增大检出率，但是也可能会出现误检情况。</li>
            <li>在出现误报的时候，可以将eMax调大。</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compRow（aclTensor）</td>
        <td>输出</td>
        <td>压缩后的行方向故障检测位流输出。</td>
        <td>
          <ul>
            <li>每个bit表示一行一段的检测结果，1表示正确，0表示检测到错误。</li>
          </ul>
        </td>
        <td>UINT8</td>
        <td>ND</td>
        <td>[ceil(M/8) * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t）</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor（aclOpExecutor）</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  其中 $splitN = \lceil N / 256\rceil$。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一阶段接口完成入参校验，出现以下场景时报错:

  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>必选输入、输出或者必选属性是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>a、b、c、weight、eMax、beSplitFactor、compRow的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>a和b的维度不为2。</td>
    </tr>
    <tr>
      <td>a的第1维（K）与b的第0维（K）不相等。</td>
    </tr>
    <tr>
      <td>eMax为负数。</td>
    </tr>
    <tr>
      <td>reduceCores小于0。</td>
    </tr>
  </tbody></table>

## aclnnMatmulAbftVerify

- **参数说明：**
    <table>
    <thead>
      <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
      <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMatmulAbftVerifyGetWorkspaceSize获取。</td></tr>
      <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
      <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnMatmulAbftVerify默认确定性实现。
- 输入矩阵a、b和c必须为2维，shape分别为[M, K]、[K, N]和[M, N]，且a的第1维（K）与b的第0维（K）必须相等。
- 输入向量weight的shape必须为[N]。
- 支持的数据类型组合为：

  | a       | b       | c       | checksumWeight      |
  |:-------:|:-------:|:-------:|:-------:|
  | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT16 |
  | BFLOAT16| BFLOAT16| FLOAT32 | BFLOAT16|
  | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |

- 不支持的场景：
  - 不支持ND格式以外的数据格式。
  - 不支持非连续的Tensor。
  - 不支持a、b和c的维度不为2的场景。

## Workspace使用设计

workspace由两部分组成：

1. 固定的16 MiB系统workspace。
2. user workspace，其中依次放置13个内部中间张量及FT内部临时区。

算子tiling阶段根据M、N、K和输入精度计算完整大小，并通过`workspaceSize`返回。调用者必须申请不少于该大小的连续device内存，不能只按`compRow`大小申请。所有中间张量的起始地址按32字节向上对齐。

定义：

```text
splitN      = ceil(N / 256)
rowSplitLen = M * splitN
bStatLen    = ceil(splitN / 8) * 8 + 8
beLen       = K * splitN
align32(x)  = ceil(x / 32) * 32
```

设`W`为user workspace首地址。在kernel内，`W = AscendC::GetUserWorkspace(workspace)`；对于算子外部的device地址计算，当前实现等价于`W = (uint8_t *)workspace + 16 MiB`。

以下偏移均相对`W`。令`S0 = 0`，每个张量的起始偏移为`Oi = align32(Si)`，结束位置为`Si+1 = Oi + 元素数 × 元素字节数`：

| 顺序 | 中间结果 | 起始地址 | 元素类型 | 元素数 |
|:--:|:--|:--|:--|--:|
| 0 | z_row | `W + O0` | FLOAT32 | `rowSplitLen` |
| 1 | d_row | `W + O1` | FLOAT32 | `rowSplitLen` |
| 2 | threshold | `W + O2` | FLOAT32 | `rowSplitLen` |
| 3 | b_mean_abs | `W + O3` | FLOAT32 | `bStatLen` |
| 4 | b_mean_square | `W + O4` | FLOAT32 | `bStatLen` |
| 5 | b_var | `W + O5` | FLOAT32 | `bStatLen` |
| 6 | be | `W + O6` | 与a相同 | `beLen` |
| 7 | be_for_aiv | `W + O7` | a为FLOAT16时是FLOAT16，否则是FLOAT32 | `beLen` |
| 8 | b_max_slice | `W + O8` | FLOAT32 | `beLen` |
| 9 | b_min_slice | `W + O9` | FLOAT32 | `beLen` |
| 10 | a_max | `W + O10` | FLOAT32 | `M` |
| 11 | a_mean | `W + O11` | FLOAT32 | `M` |
| 12 | a_min | `W + O12` | FLOAT32 | `M` |
其中FLOAT16和BFLOAT16元素占2字节，FLOAT32元素占4字节。第13段之后再次按32字节对齐，剩余区域是FT内部临时区，其逻辑大小为：

```text
M * (splitN + 1) * sizeof(float)
```

AMean计算使用的常量因子`1/K`不占用workspace。tiling阶段按输入精度生成该scalar，kernel在首次计算AMean时直接用它初始化FT的L1内部缓冲区，因此该因子不能作为workspace中间结果拷出。

如果需要调试并拷出某个中间结果，应在算子执行完成且workspace尚未释放或复用时，从上表对应的`W + Oi`开始执行device-to-host拷贝，拷贝字节数为“元素数 × 元素字节数”。对外读取workspace属于调试能力，不是稳定的公开输出接口；布局发生变更时，以[matmul_abft_verify.cpp](../op_kernel/matmul_abft_verify.cpp)中`Workspace ABI`注释下的地址切分代码为准，该代码也是偏移计算的示例实现。

## 调用示例

下面以FLOAT32输入为例。示例先调用`aclnnGemm`计算`C = A × B`，再将矩阵乘结果`C`传给`aclnnMatmulAbftVerify`进行校验。

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

    // checksumWeight全为1时，对应普通的分块行校验和。
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

    // 前序矩阵乘法：C = 1.0 * A * B + 0.0 * gemmAddend。
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

    // 对前序GEMM的结果进行ABFT校验。
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
