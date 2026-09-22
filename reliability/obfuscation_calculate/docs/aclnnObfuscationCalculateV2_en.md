# aclnnObfuscationCalculateV2

## Product Support

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function Description

- API function: sends tensor x and configuration parameters (such as param and cmd) to the PMCC obfuscation engine. The CA module of the engine calls the TA module to perform tensor obfuscation processing, and returns the obfuscated tensor y with the same shape as x.

- Background: The PMCC (Privacy and Model Confidential Computing) model obfuscation feature uses the TrustZone trusted execution environment in CPU cores to isolate and store obfuscation factors, derive obfuscation masks, and perform dynamic mask addition. Based on the NPU TrustZone, the PMCC builds the model obfuscation engine CA (Client Application in the general OS) and the model obfuscation engine TA (Trusted Application in the TEE OS). To enable the model to access the model obfuscation engine TA during inference execution, the AICPU operator mechanism and the on-card localhost socket of the NPU are used for relay. This API adds the obfCoefficient parameter. In the full-specification DeepSeek version, an obfuscation coefficient is added to ensure that the model obfuscation performance meets the requirements. Input data is processed according to the obfuscation coefficient ratio.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnObfuscationCalculateV2GetWorkspaceSize" API to obtain the required workspace size and the executor that contains the operator computation flow, and then call the "aclnnObfuscationCalculateV2" API to execute the computation.

```c++
aclnnStatus aclnnObfuscationCalculateV2GetWorkspaceSize(
  int32_t          fd,
  const aclTensor *x,
  int32_t          param,
  int32_t          cmd,
  float            obfCoefficient,
  aclTensor       *y,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnObfuscationCalculateV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnObfuscationCalculateV2GetWorkspaceSize

- **Parameter Description**
  <table style="undefined;table-layout: fixed; width: 1452px"><colgroup>
    <col style="width: 174px">
    <col style="width: 121px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 213px">
    <col style="width: 115px">
    <col style="width: 169px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Shape</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>fd (int32_t)</td>
        <td>Input</td>
        <td>Socket connection descriptor.</td>
        <td>Enter fd[0] from the output of aclnnObfuscationSetupV2 during resource initialization.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x (const aclTensor*)</td>
        <td>Input</td>
        <td>Tensor to be obfuscated.</td>
        <td>Empty tensors are not supported.</td>
        <td>FLOAT, FLOAT16, INT8, BFLOAT16</td>
        <td>ND</td>
        <td>[...,H]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>param (int32_t)</td>
        <td>Input</td>
        <td>Reserved parameter field.</td>
        <td>Only 0 is supported in the current version.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cmd (int32_t)</td>
        <td>Input</td>
        <td>Obfuscation operator command ID.</td>
        <td>Only 1 is supported in the current version.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>obfCoefficient (float)</td>
        <td>Input</td>
        <td>Obfuscation coefficient used for obfuscation processing.</td>
        <td>The value range is (0.0, 1.0].</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y (aclTensor*)</td>
        <td>Output</td>
        <td>Obfuscated tensor.</td>
        <td>The data type and shape are the same as x.</td>
        <td>FLOAT, FLOAT16, INT8, BFLOAT16</td>
        <td>ND</td>
        <td>[...,H]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size that the user needs to allocate on the device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the operator executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

<!-- npu="310p" id7 -->
- <term>Atlas inference products</term>: BFLOAT16 is not supported

<!-- end id7 -->
- **Return Value**

  aclnnStatus: status code. For details, refer to [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation. An error is reported in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
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
      <td>The input x or y is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type and data format of x are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of x and y are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of x and y are inconsistent.</td>
    </tr>
  </tbody>
  </table>

## aclnnObfuscationCalculateV2

- **Parameter Description**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device side, obtained from the first-phase API aclnnObfuscationCalculateV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

    Returns the aclnnStatus status code. For details, refer to [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Precautions

- Deterministic computation:
  - aclnnObfuscationCalculateV2 uses the deterministic implementation by default.

- This API is used together with aclnnObfuscationSetupV2 to implement the PMCC model obfuscation feature. The usage is as follows:
  - Call aclnnObfuscationSetupV2 for resource initialization. This API can be called multiple times, and the last initialization takes effect.
  - Call aclnnObfuscationCalculateV2 multiple times for tensor obfuscation processing.
  - Call aclnnObfuscationSetupV2 for resource release. This API can be called only once. Alternatively, resource release can be achieved by terminating the program process instead of explicitly calling the release API.

## Invocation Sample

The following sample code is for reference only. For the specific compilation and execution process, refer to [Compilation and Execution Sample](../../../docs/en/context/compile_and_run_sample.md).

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_obfuscation_setup_v2.h"
#include "aclnnop/aclnn_obfuscation_calculate_v2.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Fixed pattern: resource initialization
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to device memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute strides for contiguous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed pattern) device/stream initialization. Refer to the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct input and output based on the API interface.
  std::vector<int64_t> fdShape = {1};
  void* fdDeviceAddr = nullptr;

  int32_t fdToClose = 0;
  int32_t dataType = 0;
  int32_t hiddenSize = 4;
  int32_t tpRank = 0;
  int32_t modelObfSeedId = 123456;
  int32_t dataObfSeedId = 654321;
  int32_t cmd = 1;
  int32_t threadNum = 4;
  float obfCoefficient = 0.5;
  aclTensor* fd = nullptr;
  std::vector<float> fdHostData = {-1};

  // Create fd aclTensor
  ret = CreateAclTensor(fdHostData, fdShape, &fdDeviceAddr, aclDataType::ACL_INT32, &fd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnObfuscationSetupV2
  ret = aclnnObfuscationSetupV2GetWorkspaceSize(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, obfCoefficient, fd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetupV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize from the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnObfuscationSetupV2
  ret = aclnnObfuscationSetupV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetupV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed pattern) Synchronize and wait for task completion
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Get the output value. Copy the result from device memory to host side. Modify based on the specific API interface.
  auto fdSize = GetShapeSize(fdShape);
  std::vector<int32_t> fdData(fdSize, 0);
  ret = aclrtMemcpy(fdData.data(), fdData.size() * sizeof(fdData[0]), fdDeviceAddr, fdSize * sizeof(int32_t),ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR : %d\n", ret); return ret);
  for (int64_t i = 0; i < fdSize; i++) {
      LOG_PRINT("fdData[%ld] is : %d\n", i, fdData[i]);
  }

  // 6. Construct input and output based on the API interface.
  std::vector<int64_t> xShape = {2, 4};
  std::vector<int64_t> yShape = {2, 4};
  void *xDeviceAddr = nullptr;
  void *yDeviceAddr = nullptr;

  int32_t fdInput = fdData[0];
  int32_t param = 4;
  aclTensor *x = nullptr;
  int32_t cmd2 = 1;
  aclTensor *y = nullptr;
  std::vector<float> xHostData = {0.86, 0.79, 0.43, 0.37, 0.51, 0.89, 0.34, 0.49};
  std::vector<float> yHostData = {0, 0, 0, 0, 0, 0, 0, 0};

  // Create x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 7. Call the CANN operator library API. Replace with the specific API.
  uint64_t workspaceSize2 = 0;
  aclOpExecutor *executor2;

  // Call the first-phase API of aclnnObfuscationCalculateV2
  ret = aclnnObfuscationCalculateV2GetWorkspaceSize(fdInput, x, param, cmd2, obfCoefficient, y, &workspaceSize2, &executor2);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationCalculateV2GetWorkspaceSize failed. ERROR : %d\n",ret); return ret);
  // Allocate device memory based on the workspaceSize from the first-phase API
  void *workspaceAddr2 = nullptr;
  if (workspaceSize2 > 0) {
      ret = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnObfuscationCalculateV2
  ret = aclnnObfuscationCalculateV2(workspaceAddr2, workspaceSize2, executor2, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationCalculateV2 failed. ERROR : %d\n", ret); return ret);
  // 8. (Fixed pattern) Synchronize and wait for task completion
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR : %d\n", ret); return ret);

  // 9. Get the output value. y represents the obfuscated data. Copy the result from device memory to host side. Modify based on the specific API interface.
  // y
  auto ySize = GetShapeSize(yShape);
  std::vector<float> yData(ySize, 0);
  ret = aclrtMemcpy(yData.data(), yData.size() * sizeof(yData[0]), yDeviceAddr, ySize * sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR : %d\n", ret); return ret);
  for (int64_t i = 0; i < ySize; i++) {
      LOG_PRINT("yData[%ld] is : %f\n", i, yData[i]);
  }

   // 10. Construct input and output based on the API interface.
  fdToClose = fdInput;
  dataType = 0;
  hiddenSize = 0;
  tpRank = 0;
  modelObfSeedId = 0;
  dataObfSeedId = 0;
  cmd = 16;
  threadNum = 4;

  // 11. Call the CANN operator library API. Replace with the specific API name.
  uint64_t workspaceSize3 = 0;
  aclOpExecutor* executor3;

  // Call the first-phase API of aclnnObfuscationSetupV2
  ret = aclnnObfuscationSetupV2GetWorkspaceSize(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, obfCoefficient, fd, &workspaceSize3, &executor3);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetupV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize from the first-phase API
  void* workspaceAddr3 = nullptr;
  if (workspaceSize3 > 0) {
      ret = aclrtMalloc(&workspaceAddr3, workspaceSize3, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnObfuscationSetupV2
  ret = aclnnObfuscationSetupV2(workspaceAddr3, workspaceSize3, executor3, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetupV2 failed. ERROR: %d\n", ret); return ret);

  // 12. (Fixed pattern) Synchronize and wait for task completion
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 13. Release aclTensor and aclScalar used by the ObfuscationCalculateV2 API. Modify based on the specific API interface.
  aclDestroyTensor(x);
  aclDestroyTensor(y);
  aclDestroyTensor(fd);

  // 14. Release device resources used by the ObfuscationCalculateV2 API. Modify based on the specific API interface.
  aclrtFree(xDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(fdDeviceAddr);

  if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
  }
  if (workspaceSize2 > 0) {
      aclrtFree(workspaceAddr2);
  }
  if (workspaceSize3 > 0) {
      aclrtFree(workspaceAddr3);
  }
  // 15. Release device resources used by the ObfuscationSetupV2 API
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
