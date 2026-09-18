# Quick Start: Based on ops-ras Repository

## Usage Notice

This guide aims to help you quickly get started with CANN and the `ops-ras` operator repository. To help you quickly understand the entire process of operator development, we will use the **AddExample** operator as a practical object. Its source code is located in `ops-ras/examples/add_example`. The operation process is as follows:

1. **[Prerequisites](../README.md)**: Complete the environment setup and source code download by referring to the project README. The process is not repeated here. For the quick start scenario, **CANNLab or Docker deployment is recommended** for simple operation.

   > **Note**: The CANNLab or Docker environment provides the latest version of the CANN package by default. If you need to experience the latest capabilities of the master branch, you can manually set up the environment.

2. **[Compilation and Running](#i-compilation-and-running)**: Compile the custom operator package and install it to achieve quick operator invocation.

3. **[Operator Development](#ii-operator-development)**: Experience the complete loop of development, compilation, and verification by modifying the existing operator Kernel.

4. **[Operator Debugging](#iii-operator-debugging)**: Master the methods of operator printing and performance collection.

5. **[Operator Verification](#iv-operator-verification)**: Learn how to modify operator example samples to verify the functional correctness of operators under different inputs.

## I. Compilation and Running

The purpose of this stage is to **quickly experience the project standard process** and verify whether the environment can successfully perform operator source code compilation, packaging, installation, and running.

> This guide uses single operator compilation as an example. Compiling the entire operator library, offline compilation, and other scenarios are also supported. For more build parameter descriptions, see [build Parameter Description](en/install/build.md). For common issues during compilation, see [Source Code Build](en/install/compile.md).

### 1. Enter the Project Source Code

- CANNLab Cloud Development Environment:

  The latest CANN package matching project source code is provided by default. Enter the source code directory and replace `${gitCode_id}` with the developer's personal gitCode account.

  ```bash
  cd /mnt/workspace/gitCode/${gitCode_id}/ops-ras
  ```

- Non-CANNLab Cloud Development Environment:

  According to the correspondence between source code and CANN versions in the [release repository](https://gitcode.com/cann/release-management), execute the following command to download the source code. Replace `${tag_version}` with the target branch tag, for example, 9.0.0.

  ```bash
  git clone -b ${tag_version} https://gitcode.com/cann/ops-ras.git && cd ops-ras
  ```

> Note: If you need to switch the source code branch version, refer to the following guidance.
>
> 1. Execute `git branch` in the source code directory to query the current source code version.
> 2. Execute `git checkout ${tag_version}` in the source code directory to switch to the target branch source code. Ensure that the source code matches the CANN version. If the source code already exists, execute `git pull` to pull the latest source code.

### 2. Compile the AddExample Operator

The general command format: `bash build.sh --pkg --soc=<chip version> --ops=<operator name>`.

Taking the AddExample operator as an example, the compilation command is as follows:

```bash
bash build.sh --pkg --soc=${soc_version} --ops=add_example -j16
```

For the value of `${soc_version}`, visit the [CANN Download Center](https://www.hiascend.com/cann/download) and query the hardware product name according to the page prompts. The corresponding `${soc_version}` values for product names are as follows. Please pass the parameter according to the actual scenario.

- Atlas A2 Training Series Products/Atlas A2 Inference Series Products: `ascend910b`
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products: `ascend910_93`
- Ascend 950 Series Products: `ascend950`

If the following information is prompted, the compilation is successful.

```bash
Self-extractable archive "cann-ops-ras-custom_linux-${arch}.run" successfully created.
```

After successful compilation, the run package is stored in the build_out directory under the project root directory.

### 3. Install the AddExample Operator Package

```bash
./build_out/cann-ops-ras-*linux*.run
```

`AddExample` is installed in the ```${ASCEND_HOME_PATH}/opp/vendors``` path. ```${ASCEND_HOME_PATH}``` indicates the CANN software installation directory.

### 4. Configure Environment Variables

Add the path of the custom operator package to the environment variables to ensure that it can be found at runtime.

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_ras/op_api/lib:${LD_LIBRARY_PATH}
```

### 5. Quick Verification: Run Operator Sample

The general running command format: `bash build.sh --run_example <operator name> <running mode> <package mode>`.

Taking AddExample as an example, it provides a simple operator sample `add_example/examples/test_aclnn_add_example.cpp`. Run this sample to verify whether the operator function is normal.

```bash
bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
```

> **Note**: When running the sample, ensure that the `--soc` parameter is consistent with the `--soc` value used when compiling the operator package. Otherwise, `error 161001` may be reported (for example, `aclnnXxxGetWorkspaceSize failed`). If this error occurs, go back to [Section 2](#2-compile-the-addexample-operator) to check the `--soc` value, and then recompile and reinstall.

Expected output: Print the addition calculation result of the operator `AddExample`, indicating that the operator has been successfully deployed and executed correctly.

```bash
mean result[0] is: 2.000000
mean result[1] is: 2.000000
mean result[2] is: 2.000000
mean result[3] is: 2.000000
mean result[4] is: 2.000000
mean result[5] is: 2.000000
mean result[6] is: 2.000000
mean result[7] is: 2.000000
...
```

## II. Operator Development

The purpose of this stage is to try **modifying the kernel function code** for the successfully running AddExample operator.

### 1. Modify Kernel Implementation

Find the core kernel implementation file of the AddExample operator `ops-ras/examples/add_example/op_kernel/add_example.h`, and try to change the Add operation in the operator to a Mul operation:

```cpp
__aicore__ inline void AddExample<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    // === Replace Add with Mul here ===
    // AscendC::Add(zLocal, xLocal, yLocal, tileLength_);
    AscendC::Mul(zLocal, xLocal, yLocal, tileLength_);
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}
```

### 2. Compile and Verify

Repeat the steps in the [Compilation and Running](#i-compilation-and-running) section:

1. **Recompile**:

    First return to the project root directory. The compilation command is as follows:

    ```bash
    bash build.sh --pkg --soc=${soc_version} --ops=add_example -j16
    ```

    > **Note**: Please fill in `${soc_version}` according to the actual chip model. The value method is the same as described in [Compile the AddExample Operator](#2-compile-the-addexample-operator).

2. **Reinstall**:

    ```bash
    ./build_out/cann-ops-ras-*linux*.run
    ```

3. **Re-verify**:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
    ```

4. **Success Sign**: The output result becomes the multiplication result.

    ```bash
    mean result[0] is: 1.000000
    mean result[1] is: 1.000000
    mean result[2] is: 1.000000
    mean result[3] is: 1.000000
    mean result[4] is: 1.000000
    mean result[5] is: 1.000000
    mean result[6] is: 1.000000
    mean result[7] is: 1.000000
    ...
    ```

## III. Operator Debugging

This stage takes AddExample as an example to add printing in the operator and collect operator performance data for subsequent problem analysis and positioning.

### 1. Printing

If the operator has execution failure, precision abnormality, or other problems, add printing for problem analysis and positioning.

Please modify the code in `examples/add_example/op_kernel/add_example.h`.

* **printf**

  This interface supports printing Scalar type data, such as integers, character types, Boolean types, and so on. For detailed introduction, see "Operator Debugging API > printf" in "[Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi)".

  ```c++
  blockLength_ = (tilingData->totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) ?
        ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) : 1;
  // Print the current kernel calculation Block length
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```

* **DumpTensor**

  This interface supports dumping the content of the specified Tensor, and also supports printing custom additional information, such as the current line number. For detailed introduction, see "Operator Debugging API > DumpTensor" in "[Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi)".

  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // Print zLocal Tensor information
  DumpTensor(zLocal, 0, 128);
  ```

### 2. Performance Collection

When the operator function verification is correct, you can collect operator performance data through the `msprof` tool.

- **Generate Executable File**

    Call the example sample of the AddExample operator to generate an executable file (test_aclnn_add_example), which is located in the project `ops-ras/build` directory.

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
    ```

- **Collect Performance Data**

    Enter the AddExample operator executable file directory `ops-ras/build/` and execute the following command:

    ```bash
    msprof --application="./test_aclnn_add_example"
    ```

The collection result is in the project `ops-ras/build/` directory. After the msprof command is executed, it will automatically parse and export the performance data result file. For detailed content, see [msprof](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/T&ITools/Profiling/atlasprofiling_16_0110.html#ZH-CN_TOPIC_0000002504160251).

## IV. Operator Verification

This stage verifies the functional correctness of the operator in multiple scenarios by modifying the input data of the AddExample operator example sample.

### 1. Modify Test Input

Find and edit the `ops-ras/examples/add_example/examples/test_aclnn_add_example.cpp` of `AddExample`, and modify the shape and numerical values of the input tensor.

**Modify Input/Output Data**: Modify the shape information of input and output, as well as the initialization data, and construct the corresponding input and output tensors.

```c++
int main() {
    // ... initialization code ...

    // === ① Modify selfX input ===
    // Before modification: shape = {32, 4, 4, 4}, all values are 1
    // After modification: change input shape to {8, 8, 8, 8}, and fill with different test data
    std::vector<int64_t> selfXShape = {8, 8, 8, 8};
    std::vector<float> selfXHostData(4096); // 4096 = 8 * 8 * 8 *8
    // You can use a loop to fill more distinguishable data, such as an increasing sequence
    for (int i = 0; i < 4096; ++i) {
        selfXHostData[i] = static_cast<float>(i % 10); // Fill with cyclic values of 0-9
    }
    // === ② Refer to selfX, similarly modify selfY and selfZ inputs ===

    // ... subsequent execution code ...
}
```

### 2. Recompile and Verify

1. Since only the example test code is modified, there is no need to recompile the operator package.

2. Re-execute the verification command:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
    ```

3. Observe whether the operator output result meets expectations.

## Conclusion

After experiencing the above processes, you have basically completed the operator development process. If you want to further contribute new operators or learn more advanced development, debugging, and other skills, please visit the project README to learn about [Advanced Tutorials](../README.md#学习教程) and [Contribution Guide](../README.md#相关信息), and so on.