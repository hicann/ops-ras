# 算子接口（aclnn）

## 使用说明

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

调用算子API时，需引用依赖的头文件和库文件，一般头文件默认在`${INSTALL_DIR}/include/aclnnop`，库文件默认在`${INSTALL_DIR}/lib64`，具体文件如下：

- 依赖的头文件：①方式1 （推荐）：引用算子总头文件aclnn\_ops\_\$\{ops\_project\}.h。②方式2：按需引用单算子API头文件aclnn\_\*.h。
- 依赖的库文件：按需引用算子总库文件libopapi\_\$\{ops\_project\}.so。

其中${INSTALL_DIR}表示CANN安装后文件路径；\$\{ops\_project\}表示算子仓（如math、nn、cv、transformer），请配置为实际算子仓名。

## 接口列表

> **确定性简介**：
>
> - 配置说明：因CANN或NPU型号不同等原因，可能无法保证同一个算子多次运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分算子接口可通过`aclrtCtxSetSysParamOpt`（参见[《acl API（C）》](https://hiascend.com/document/redirect/CannCommunityCppApi)）开启确定性算法，使多次运行结果一致。
> - 性能说明：同一个算子采用确定性计算通常比非确定性慢，因此模型单次运行性能可能会下降。但在实验、调试调测等需要保证多次运行结果相同来定位问题的场景，确定性计算可以提升效率。
> - 线程说明：同一线程中只能设置一次确定性状态，多次设置以最后一次有效设置为准。有效设置是指设置确定性状态后，真正执行了一次算子任务下发。如果仅设置，没有算子下发，只能是确定性变量开启但未下发给算子，因此不执行算子。
>   解决方案：暂不推荐一个线程多次设置确定性。该问题在二进制开启和关闭情况下均存在，在后续版本中会解决该问题。

算子接口列表如下：

|    接口名   |      说明     |    确定性说明（A2/A3）    |    确定性说明（A5）    |
|-----------|------------|------------|------------|
| [aclnnObfuscationCalculate](../../reliability/obfuscation_calculate/docs/aclnnObfuscationCalculate.md) | 将张量x和配置参数（如param、cmd）发送至PMCC混淆引擎。引擎的CA模块调用TA模块，进行张量混淆处理，最终返回shape与x一致的混淆后的张量y。 |默认确定性实现| - |
| [aclnnObfuscationCalculateV2](../../reliability/obfuscation_calculate/docs/aclnnObfuscationCalculateV2.md) | 将张量x和配置参数（如param、cmd）发送至PMCC混淆引擎。引擎的CA模块调用TA模块，进行张量混淆处理，最终返回shape与x一致的混淆后的张量y。 |默认确定性实现| - |
| [aclnnObfuscationSetup](../../reliability/obfuscation_setup/docs/aclnnObfuscationSetup.md) | 完成PMCC模型混淆引擎的资源初始化和释放。 |默认确定性实现| - |
| [aclnnObfuscationSetupV2](../../reliability/obfuscation_setup/docs/aclnnObfuscationSetupV2.md) | 完成PMCC模型混淆引擎的资源初始化和释放。 |默认确定性实现| - |
| [aclnnCrypto](../../reliability/crypto/docs/aclnnCrypto.md) | 调用 aicpu 加解密算子，按输入参数执行。 |默认确定性实现| - |
