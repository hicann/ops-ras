# aclnnObfuscationSetup

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

- API function: initializes and releases resources for the PMCC model obfuscation engine.

   - Resource initialization: establishes a socket connection with the PMCC obfuscation engine CA, initializes the CA and TA, and returns the socket connection descriptor.
   - Resource release: disconnects the socket connection with the PMCC obfuscation engine CA.

- Background: The PMCC (Privacy and Model Confidential Computing) model obfuscation feature uses the TrustZone trusted execution environment in CPU cores to isolate and store obfuscation factors, derive obfuscation masks, and perform dynamic mask addition. Based on the NPU TrustZone, the PMCC builds the model obfuscation engine CA (Client Application in the general OS) and the model obfuscation engine TA (Trusted Application in the TEE OS). To enable the model to access the model obfuscation engine TA during inference execution, the AICPU operator mechanism and the on-card localhost socket of the NPU are used for relay.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnObfuscationSetupGetWorkspaceSize" API to obtain the required workspace size and the executor that contains the operator computation flow, and then call the "aclnnObfuscationSetup" API to execute the computation.

```c++
aclnnStatus aclnnObfuscationSetupGetWorkspaceSize(
  int32_t         fdToClose,
  int32_t         dataType,
  int32_t         hiddenSize,
  int32_t         tpRank,
  int32_t         modelObfSeedId,
  int32_t         dataObfSeedId,
  int32_t         cmd,
  int32_t         threadNum,
  aclTensor      *fd,
  uint64_t       *workspaceSize,
  aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnObfuscationSetup(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnObfuscationSetupGetWorkspaceSize

- **Parameter Description**

  <table style="undefined;table-layout: fixed; width: 1452px"><colgroup>
    <col style="width: 174px">
    <col style="width: 121px">
    <col style="width: 253px">
    <col style="width: 361px">
    <col style="width: 213px">
    <col style="width: 110px">
    <col style="width: 110px">
    <col style="width: 110px">
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
        <td>fdToClose (int32_t)</td>
        <td>Input</td>
        <td>Socket connection descriptor to be closed.</td>
        <td>When cmd is set to 3, enter the fd returned by this operator when cmd is set to 1. Otherwise, enter 0.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dataType (int32_t)</td>
        <td>Input</td>
        <td>ID representing the tensor data type.</td>
        <td>
          <ul>
            <li>A valid value is required only when cmd is set to 1 or 2. Otherwise, enter 0.</li>
            <li><term>Atlas inference products</term>: Select from {0, 1}. 0 indicates FLOAT and 1 indicates FLOAT16.</li>
            <li><term>Atlas A2 series products</term>: Select from {0, 1, 2, 27}. 0 indicates FLOAT, 1 indicates FLOAT16, 2 indicates INT8, and 27 indicates BF16.</li>
          </ul>
        </td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hiddenSize (int32_t)</td>
        <td>Input</td>
        <td>Hidden layer dimension.</td>
        <td>A valid value is required only when cmd is set to 1 or 2. Otherwise, enter 0. The value range is 1 to 10000.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>tpRank (int32_t)</td>
        <td>Input</td>
        <td>TP Rank.</td>
        <td>The value range is 0 to 1024. A valid value is required only when cmd is set to 1 or 2. Otherwise, enter 0.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>modelObfSeedId (int32_t)</td>
        <td>Input</td>
        <td>Model obfuscation factor ID.</td>
        <td>Used by the TA to query the model obfuscation factor from the TEE KMC. A valid value is required only when cmd is set to 1 or 2. Otherwise, enter 0.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dataObfSeedId (int32_t)</td>
        <td>Input</td>
        <td>Data obfuscation factor ID.</td>
        <td>Used by the TA to query the data obfuscation factor from the TEE KMC. A valid value is required only when cmd is set to 1 or 2. Otherwise, enter 0.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cmd (int32_t)</td>
        <td>Input</td>
        <td>Setup command ID.</td>
        <td>Select from {1, 2, 16}. When set to 1, resource initialization is performed in normal mode. When set to 2, resource initialization is performed in high-precision mode. When set to 16, resource release is performed.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>threadNum (int32_t)</td>
        <td>Input</td>
        <td>Number of threads used by the CA/TA for obfuscation processing.</td>
        <td>Select from {1, 2, 3, 4, 5, 6}. A valid value is required only when cmd is set to 1 or 2. Otherwise, enter 0.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>fd (aclTensor*)</td>
        <td>Output</td>
        <td>Socket connection descriptor.</td>
        <td>Empty tensors are not supported.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
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
      <td>The input fd is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type and data format of fd are not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnObfuscationSetup

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
      <td>Workspace size allocated on the device side, obtained from the first-phase API aclnnObfuscationSetupGetWorkspaceSize.</td>
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
  - aclnnObfuscationSetup uses the deterministic implementation by default.

- This API is used together with aclnnObfuscationCalculate to implement the PMCC model obfuscation feature. The usage is as follows:
  - Call aclnnObfuscationSetup for resource initialization. This API can be called multiple times, and the last initialization takes effect.
  - Call aclnnObfuscationCalculate multiple times for tensor obfuscation processing.
  - Call aclnnObfuscationSetup for resource release. This API can be called only once. Alternatively, resource release can be achieved by terminating the program process instead of explicitly calling the release API.

## Invocation Sample

The following sample code is for reference only. For the specific compilation and execution process, refer to [Compilation and Execution Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  aclTensor* fd = nullptr;
  std::vector<float> fdHostData = {-1};

  // Create fd aclTensor
  ret = CreateAclTensor(fdHostData, fdShape, &fdDeviceAddr, aclDataType::ACL_INT32, &fd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnObfuscationSetup
  ret = aclnnObfuscationSetupGetWorkspaceSize(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, fd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetupGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize from the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnObfuscationSetup
  ret = aclnnObfuscationSetup(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetup failed. ERROR: %d\n", ret); return ret);

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

  // Call the first-phase API of aclnnObfuscationCalculate
  ret = aclnnObfuscationCalculateGetWorkspaceSize(fdInput, x, param, cmd2, y, &workspaceSize2, &executor2);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationCalculateGetWorkspaceSize failed. ERROR : %d\n",ret); return ret);
  // Allocate device memory based on the workspaceSize from the first-phase API
  void *workspaceAddr2 = nullptr;
  if (workspaceSize2 > 0) {
      ret = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnObfuscationCalculate
  ret = aclnnObfuscationCalculate(workspaceAddr2, workspaceSize2, executor2, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationCalculate failed. ERROR : %d\n", ret); return ret);
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

  // Call the first-phase API of aclnnObfuscationSetup
  ret = aclnnObfuscationSetupGetWorkspaceSize(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, fd, &workspaceSize3, &executor3);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetupGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize from the first-phase API
  void* workspaceAddr3 = nullptr;
  if (workspaceSize3 > 0) {
      ret = aclrtMalloc(&workspaceAddr3, workspaceSize3, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnObfuscationSetup
  ret = aclnnObfuscationSetup(workspaceAddr3, workspaceSize3, executor3, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetup failed. ERROR: %d\n", ret); return ret);

  // 12. (Fixed pattern) Synchronize and wait for task completion
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 13. Release aclTensor and aclScalar used by the ObfuscationCalculate API. Modify based on the specific API interface.
  aclDestroyTensor(x);
  aclDestroyTensor(y);
  aclDestroyTensor(fd);

  // 14. Release device resources used by the ObfuscationCalculate API. Modify based on the specific API interface.
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
  // 15. Release device resources used by the ObfuscationSetup API
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
