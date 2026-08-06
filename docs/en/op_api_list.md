# Operator APIs (aclnn)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-30T03:53:18.085Z pushedAt=2026-07-30T03:55:25.192Z -->

## Instructions

To facilitate operator calling, a set of C-based APIs (prefixed with aclnn) is provided, eliminating the need for IR (Intermediate Representation) definitions and enabling efficient model building and app development. This approach is referred to as "single-operator API call", abbreviated as aclnn call.

When calling an operator API, you need to reference the dependent header files and library files. Generally, header files are located in `${INSTALL_DIR}/include/aclnnop` by default, and library files are located in `${INSTALL_DIR}/lib64` by default. The specific files are as follows:

- Dependent header files: ① Method 1 (recommended): Reference the operator master header file aclnn_ops_${ops_project}.h. ② Method 2: Reference the single-operator API header file aclnn_*.h as needed.

- Dependent library file: Reference the operator master library file libopapi_${ops_project}.so as needed.

Here, ${INSTALL_DIR} indicates the CANN installation directory, and \$\{ops\_project\} indicates the operator repository (such as math, nn, cv, transformer). Set it to the actual operator repository name.

## API List

> **Determinism Overview**:
>
> - Configuration: Due to factors such as different CANN or NPU models, it may not be possible to guarantee consistent results across multiple runs of the same operator. Under the same conditions (platform, device, version, and other randomness parameters), some operator APIs can enable deterministic algorithms through `aclrtCtxSetSysParamOpt` (see [*acl API (C)*](https://hiascend.com/document/redirect/CannCommunityCppApi)) to ensure consistent results across multiple runs.
> - Performance: Deterministic computation for the same operator is typically slower than non-deterministic computation, which may degrade the single-run performance of the model. However, in scenarios such as experimentation, debugging, and troubleshooting where consistent results across multiple runs are required to locate issues, deterministic computation can improve efficiency.
> - Threading: The deterministic state can be set only once per thread. If set multiple times, only the last valid setting takes effect. A valid setting means that after the deterministic state is set, an operator task dispatch is actually executed. If only the setting is performed without any operator dispatch, the deterministic variable is enabled but not dispatched to the operator, so the operator is not executed.
>   Solution: Setting determinism multiple times within a single thread is not recommended at this time. This issue exists in both binary enabled and disabled scenarios and will be resolved in a future version.

The operator API list is as follows:

|    API Name   |      Description     |    Deterministic Implementation (A2/A3)    |    Deterministic Implementation (A5)    |
|----------- |------------ |------------|------------|
| [aclnnObfuscationCalculate](../../reliability/obfuscation_calculate/docs/aclnnObfuscationCalculate_en.md) | Sends the tensor x and configuration parameters (such as param and cmd) to the PMCC obfuscation engine. The CA module of the engine invokes the TA module to perform tensor obfuscation, and finally returns an obfuscated tensor y with the same shape as x. |Default|
| [aclnnObfuscationCalculateV2](../../reliability/obfuscation_calculate/docs/aclnnObfuscationCalculateV2_en.md) | Sends the tensor x and configuration parameters (such as param and cmd) to the PMCC obfuscation engine. The CA module of the engine invokes the TA module to perform tensor obfuscation, and finally returns an obfuscated tensor y with the same shape as x. |Default|
| [aclnnObfuscationSetup](../../reliability/obfuscation_setup/docs/aclnnObfuscationSetup_en.md) | Completes resource initialization and release for the PMCC model obfuscation engine. |Default|
| [aclnnObfuscationSetupV2](../../reliability/obfuscation_setup/docs/aclnnObfuscationSetupV2_en.md) | Completes resource initialization and release for the PMCC model obfuscation engine. |Default|
| [aclnnCrypto](../../reliability/crypto/docs/aclnnCrypto_en.md) | Invokes the aicpu encryption/decryption operator and executes based on the input parameters. | Default |
