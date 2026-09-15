# 算子接口（aclnn）

## 使用说明

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

调用算子API时，需引用依赖的头文件和库文件，一般头文件默认在`${INSTALL_DIR}/include/aclnnop`，库文件默认在`${INSTALL_DIR}/lib64`，具体文件如下：

- 依赖的头文件：①方式1 （推荐）：引用算子总头文件aclnn\_ops\_\$\{ops\_project\}.h。②方式2：按需引用单算子API头文件aclnn\_\*.h。
- 依赖的库文件：按需引用算子总库文件libopapi\_\$\{ops\_project\}.so。注意，原所有算子仓总库文件libopapi.so后续会废弃，不推荐使用，也不支持与单个算子仓库文件同时使用。

其中${INSTALL_DIR}表示CANN安装后文件路径；\$\{ops\_project\}表示算子仓名（如ras、nn、math等），请配置为实际算子仓名。

- **V版本演进说明**

  请注意，部分API存在多个V版本，使用时选择最高V版本即可（高版本API已兼容低版本API的所有能力）。

## 接口列表

> [!NOTE]
>
> - 算子特性介绍：调用API前，请先学习算子相关基础知识，包括**确定性算法**、**Batch一致性**、**常见量化模式**等，具体介绍参见[算子基本概念](context/basic_concept.md)。
> - 符号说明：表格中“-”符号表示该接口暂不支持当前列产品。

算子接口列表如下：

|    接口名   |      说明     |    确定性说明（A2/A3）    |    确定性说明（Ascend 950）    |
|-----------|------------|------------|------------|
| [aclnnObfuscationCalculate](../../reliability/obfuscation_calculate/docs/aclnnObfuscationCalculate.md) | 将张量x和配置参数（如param、cmd）发送至PMCC混淆引擎。引擎的CA模块调用TA模块，进行张量混淆处理，最终返回shape与x一致的混淆后的张量y。 |默认确定性实现| - |
| [aclnnObfuscationCalculateV2](../../reliability/obfuscation_calculate/docs/aclnnObfuscationCalculateV2.md) | 将张量x和配置参数（如param、cmd）发送至PMCC混淆引擎。引擎的CA模块调用TA模块，进行张量混淆处理，最终返回shape与x一致的混淆后的张量y。 |默认确定性实现| - |
| [aclnnObfuscationSetup](../../reliability/obfuscation_setup/docs/aclnnObfuscationSetup.md) | 完成PMCC模型混淆引擎的资源初始化和释放。 |默认确定性实现| - |
| [aclnnObfuscationSetupV2](../../reliability/obfuscation_setup/docs/aclnnObfuscationSetupV2.md) | 完成PMCC模型混淆引擎的资源初始化和释放。 |默认确定性实现| - |
| [aclnnCrypto](../../reliability/crypto/docs/aclnnCrypto.md) | 调用aicpu加解密算子，按输入参数执行。 |默认确定性实现| - |
