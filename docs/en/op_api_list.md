# Operator Interfaces (aclnn)

## Usage Instructions

To facilitate operator invocation, a set of C-based APIs (prefixed with aclnn) is provided. These APIs do not require IR (Intermediate Representation) definitions, making it convenient and efficient to build models and develop applications. This method is called "single operator API invocation," abbreviated as aclnn invocation.

When calling operator APIs, you need to reference the dependent header files and library files. Generally, the header files are located by default at `${INSTALL_DIR}/include/aclnnop`, and the library files are located by default at `${INSTALL_DIR}/lib64`. The specific files are as follows:

- Dependent header files: (1) Method 1 (Recommended): Reference the operator aggregate header file `aclnn_ops_${ops_project}.h`. (2) Method 2: Reference individual operator API header files `aclnn_*.h` as needed.
- Dependent library files: Reference the operator aggregate library file `libopapi_${ops_project}.so` as needed. Note that the original aggregate library file `libopapi.so` for all operator repositories will be deprecated in the future. It is not recommended for use, and it does not support simultaneous use with individual operator repository files.

Here, `${INSTALL_DIR}` represents the path where CANN is installed, and `${ops_project}` represents the operator repository name (such as ras, nn, math). Configure it to the actual operator repository name.

- **V Version Evolution Description**

  Note that some APIs have multiple V versions. When using them, select the highest V version (higher-version APIs already include all the capabilities of lower-version APIs).

## Interface List

> **Determinism Description**:
>
> - Configuration description: Due to factors such as different CANN or NPU models, it may not be possible to guarantee that multiple runs of the same operator produce consistent results. Under the same conditions (platform, device, version number, and other randomness parameters), some operator interfaces can enable deterministic algorithms through `aclrtCtxSetSysParamOpt` (see [Runtime API](https://hiascend.com/document/redirect/CannCommunityCppApi)) to make multiple runs produce consistent results.
> - Performance description: The same operator using deterministic computation is typically slower than non-deterministic computation, so the single-run performance of the model may degrade. However, in scenarios such as experimentation, debugging, and troubleshooting that require consistent results across multiple runs to locate issues, deterministic computation can improve efficiency.
> - Thread description: The deterministic state can only be set once in the same thread. If set multiple times, the last effective setting takes precedence. An effective setting means that after setting the deterministic state, an operator task has actually been dispatched. If only the setting is made without dispatching an operator, the deterministic variable is enabled but not dispatched to the operator, so the operator is not executed.
>   Solution: It is not recommended to set determinism multiple times in one thread. This issue exists in both binary-enabled and binary-disabled cases and will be resolved in a future version.
> - Symbol description: The "-" symbol in the table indicates that the interface does not currently support the product in that column.

The operator interface list is as follows:

|    Interface Name   |      Description     |    Determinism Description (A2/A3)    |    Determinism Description (Ascend 950)    |
|-----------|------------|------------|------------|
| [aclnnObfuscationCalculate](../../reliability/obfuscation_calculate/docs/aclnnObfuscationCalculate.md) | Sends the tensor x and configuration parameters (such as param, cmd) to the PMCC obfuscation engine. The CA module of the engine calls the TA module to perform tensor obfuscation processing, and finally returns an obfuscated tensor y with the same shape as x. |Default deterministic implementation| - |
| [aclnnObfuscationCalculateV2](../../reliability/obfuscation_calculate/docs/aclnnObfuscationCalculateV2.md) | Sends the tensor x and configuration parameters (such as param, cmd) to the PMCC obfuscation engine. The CA module of the engine calls the TA module to perform tensor obfuscation processing, and finally returns an obfuscated tensor y with the same shape as x. |Default deterministic implementation| - |
| [aclnnObfuscationSetup](../../reliability/obfuscation_setup/docs/aclnnObfuscationSetup.md) | Completes the resource initialization and release of the PMCC model obfuscation engine. |Default deterministic implementation| - |
| [aclnnObfuscationSetupV2](../../reliability/obfuscation_setup/docs/aclnnObfuscationSetupV2.md) | Completes the resource initialization and release of the PMCC model obfuscation engine. |Default deterministic implementation| - |
| [aclnnCrypto](../../reliability/crypto/docs/aclnnCrypto.md) | Calls the aicpu encryption/decryption operator and executes according to the input parameters. |Default deterministic implementation| - |