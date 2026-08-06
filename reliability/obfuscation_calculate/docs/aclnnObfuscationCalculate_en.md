# aclnnObfuscationCalculate

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-30T01:47:45.560Z pushedAt=2026-07-30T03:35:18.426Z -->

## Applicable Products

| Product                                                        | Supported |
| :------------------------------------------------------------- | :-------: |
| Ascend 950PR/Ascend 950DT                                      |     ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term> |     ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term> |     √     |
| <term>Atlas 200I/500 A2 inference products</term>              |     ×     |
| <term>Atlas inference products</term>                          |     √     |
| <term>Atlas training products</term>                           |     ×     |

## Function

- Description: Sends the tensor x and configuration parameters (such as param and cmd) to the PMCC obfuscation engine. The CA module of the engine calls the TA module to perform tensor obfuscation, and finally returns an obfuscated tensor y with the same shape as x.

- Background: The PMCC (Privacy & Model Confidential Computing) model obfuscation feature uses the TrustZone trusted execution environment in the CPU core to isolate and store obfuscation factors, derive obfuscation masks, and perform dynamic mask addition. PMCC builds the model obfuscation engine CA (Client Application in the normal OS) and the model obfuscation engine TA (Trusted Application in the TEE OS) based on the NPU TrustZone. To enable the model to access the model obfuscation engine TA during inference execution, the AICPU operator mechanism and the localhost socket within the NPU card are used for relay.

## Function Prototype

Each operator is divided into a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnObfuscationCalculateGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnObfuscationCalculate" API to perform the computation.

```c++
aclnnStatus aclnnObfuscationCalculateGetWorkspaceSize(
  int32_t          fd,
  const aclTensor *x,
  int32_t          param,
  int32_t          cmd,
  aclTensor       *y,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnObfuscationCalculate(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnObfuscationCalculateGetWorkspaceSize

- **Parameters**

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
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>fd (int32_t)</td>
        <td>Input</td>
        <td>Socket connector.</td>
        <td>The data type is INT32.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x（const aclTensor*）</td>
        <td>Input</td>
        <td>The tensor to be obfuscated.</td>
        <td>Empty tensors are not supported.</td>
        <td>FLOAT, FLOAT16, INT8, BFLOAT16</td>
        <td>ND</td>
        <td>[...,H]</td>
        <td>×</td>
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
        <td>Obfuscation operator instruction number.</td>
        <td>Only 1 is supported in the current version.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y (aclTensor*)</td>
        <td>Output</td>
        <td>The obfuscated tensor.</td>
        <td>The data type and shape are the same as those of x.</td>
        <td>FLOAT, FLOAT16, INT8, BFLOAT16</td>
        <td>ND</td>
        <td>[...,H]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size that the user needs to allocate on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- <term>Atlas inference products</term>: BFLOAT16 is not supported.

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

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

## aclnnObfuscationCalculate

- **Parameters**

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained from the first-phase API aclnnObfuscationCalculateGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  - aclnnObfuscationCalculate defaults to a deterministic implementation.

- This API is used together with [aclnnObfuscationSetup](../../obfuscation_setup/docs/aclnnObfuscationSetup_en.md) to implement the PMCC model obfuscation function. The usage is as follows:

  - First, call aclnnObfuscationSetup for resource initialization. It can be called repeatedly, and the last initialization takes effect.

  - Then, call aclnnObfuscationCalculate multiple times for tensor obfuscation.

  - Finally, call aclnnObfuscationSetup for resource release, which can only be called once. Alternatively, resource release can be achieved by terminating the program process instead of explicitly calling it.

## Example

The sample code is as follows for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_obfuscation_setup.h"
#include "aclnnop/aclnn_obfuscation_calculate.h"

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
  // Boilerplate, resource initialization.
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
  // Call aclrtMalloc to apply for device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the ACL API manual.
  // Fill in the deviceId based on your actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API interface.
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
  aclTensor* fd = nullptr;
  std::vector<float> fdHostData = {-1};

  //Create the fd aclTensor.
  ret = CreateAclTensor(fdHostData, fdShape, &fdDeviceAddr, aclDataType::ACL_INT32, &fd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the aclnnObfuscationSetup first-phase API.
  ret = aclnnObfuscationSetupGetWorkspaceSize(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, fd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetupGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Apply for device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the aclnnObfuscationSetup second-phase API.
  ret = aclnnObfuscationSetup(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetup failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side memory to the host side. Modify this based on the specific API definition.
  auto fdSize = GetShapeSize(fdShape);
  std::vector<int32_t> fdData(fdSize, 0);
  ret = aclrtMemcpy(fdData.data(), fdData.size() * sizeof(fdData[0]), fdDeviceAddr, fdSize * sizeof(int32_t),ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR : %d\n", ret); return ret);
  for (int64_t i = 0; i < fdSize; i++) {
      LOG_PRINT("fdData[%ld] is : %d\n", i, fdData[i]);
  }

  // 6. Construct the input and output. Construct them based on the API definition.
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

  // Create x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create y aclTensor.
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 7. Call the CANN operator library API. Replace with the specific API.
  uint64_t workspaceSize2 = 0;
  aclOpExecutor *executor2;

  // Call the first-phase API of aclnnObfuscationCalculate.
  ret = aclnnObfuscationCalculateGetWorkspaceSize(fdInput, x, param, cmd2, y, &workspaceSize2, &executor2);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationCalculateGetWorkspaceSize failed. ERROR : %d\n",ret); return ret);
  // Apply for device memory based on the workspaceSize calculated by the first-phase API.
  void *workspaceAddr2 = nullptr;
  if (workspaceSize2 > 0) {
      ret = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnObfuscationCalculate.
  ret = aclnnObfuscationCalculate(workspaceAddr2, workspaceSize2, executor2, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationCalculate failed. ERROR : %d\n", ret); return ret);
  // 8. Boilerplate: wait synchronously for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR : %d\n", ret); return ret);

  // 9. Obtain the output value. y represents the obfuscated data. Copy the result from the device memory to the host memory. Modify based on the specific API definition.
  // y
  auto ySize = GetShapeSize(yShape);
  std::vector<float> yData(ySize, 0);
  ret = aclrtMemcpy(yData.data(), yData.size() * sizeof(yData[0]), yDeviceAddr, ySize * sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR : %d\n", ret); return ret);
  for (int64_t i = 0; i < ySize; i++) {
      LOG_PRINT("yData[%ld] is : %f\n", i, yData[i]);
  }

   // 10. Construct the input and output. Customize the construction based on the API definition.
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

  // Call the first-phase API of aclnnObfuscationSetup.
  ret = aclnnObfuscationSetupGetWorkspaceSize(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, fd, &workspaceSize3, &executor3);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetupGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Apply for device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr3 = nullptr;
  if (workspaceSize3 > 0) {
      ret = aclrtMalloc(&workspaceAddr3, workspaceSize3, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnObfuscationSetup.
  ret = aclnnObfuscationSetup(workspaceAddr3, workspaceSize3, executor3, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetup failed. ERROR: %d\n", ret); return ret);

  // 12. (Boilerplate) Synchronously wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 13. Release the aclTensor and aclScalar involved in the ObfuscationCalculate API. Modify based on the specific API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(y);
  aclDestroyTensor(fd);

  // 14. Release the device resources involved in the ObfuscationCalculate API. Modify based on the specific API definition.
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
  // 15. Release the device resources involved in the ObfuscationSetup API.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
