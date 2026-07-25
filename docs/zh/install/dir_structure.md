# 项目目录

> 本章罗列的部分目录是可选的，请以实际交付件为准。尤其**单算子目录**，不同场景下交付件有差异，具体说明如下：
>
> - 若缺少op_host目录，可能是调用了其他算子op_host实现，调用逻辑参见该算子op_api目录下源码实现；如有需要，欢迎开发者参考[贡献指南](../../../CONTRIBUTING.md)补充贡献该算子。
> - 若缺少op_api目录，说明该算子暂不支持aclnn调用。
> - 若缺少op_graph或op_kernel目录，说明该算子暂不支持图模式调用或暂无Ascend C Kernel实现。

项目全量目录层级介绍如下：

```text
├── cmake                                               # 项目工程编译目录
│   ├── modules                                         # CMake Find模块目录
│   │   ├── FindOPBASE.cmake                            # OPBASE依赖查找
│   │   ├── Findaicpu.cmake                             # aicpu依赖查找
│   │   ├── Finddlog.cmake                              # dlog依赖查找
│   │   ├── Findmetadef.cmake                           # metadef依赖查找
│   │   ├── Findnnopbase.cmake                          # nnopbase依赖查找
│   │   ├── Findplatform.cmake                          # platform依赖查找
│   │   ├── Findruntime.cmake                           # runtime依赖查找
│   │   ├── Findsecurec.cmake                           # securec依赖查找
│   │   └── Findtilingapi.cmake                         # tilingapi依赖查找
│   ├── third_party                                     # 第三方依赖CMake配置目录
│   │   ├── build                                       # 第三方依赖构建目录（含patch等）
│   │   ├── abseil-cpp.cmake                            # abseil-cpp依赖配置
│   │   ├── ascend_protobuf.cmake                       # protobuf依赖配置
│   │   ├── eigen.cmake                                 # eigen依赖配置
│   │   ├── gtest.cmake                                 # gtest依赖配置
│   │   ├── makeself-fetch.cmake                        # makeself获取配置
│   │   └── nlohmann_json.cmake                         # nlohmann_json依赖配置
│   ├── Third_Party_Open_Source_Software_List.yaml      # 第三方开源软件清单
│   ├── aclnn_ops_ras.h.in                              # aclnn汇总头文件模板
│   ├── custom_kernel.cmake                             # 自定义Kernel编译配置
│   ├── dependencies.cmake                              # 项目依赖管理
│   ├── func.cmake                                      # 公共CMake函数定义
│   ├── gen_ops_info.cmake                              # 算子信息生成脚本
│   ├── intf_pub_linux.cmake                            # Linux接口发布配置
│   ├── intf_pub_llt_gccnative.cmake                    # LLT GCC Native接口发布配置
│   ├── makeself_built_in.cmake                         # makeself内置配置
│   ├── makeself_custom.cmake                           # makeself自定义配置
│   ├── opbuild.cmake                                   # 算子构建配置
│   ├── package.cmake                                   # 打包配置
│   ├── runtimeKB.cmake                                 # 运行时知识库配置
│   ├── static.cmake                                    # 静态库编译配置
│   ├── symbol.cmake                                    # 符号导出配置
│   ├── ut.cmake                                        # UT测试编译配置
│   └── variables.cmake                                 # 全局变量定义
├── common                                              # 项目公共头文件和公共代码
│   ├── CMakeLists.txt
│   ├── act                                             # Activation相关公共实现
│   │   ├── epilogue                                    # Epilogue处理（如反量化、SwishGLU量化等）
│   │   ├── matmul                                      # 矩阵乘相关实现（block、kernel、policy、tile）
│   │   ├── prologue                                    # Prologue处理（如Cast、广播等）
│   │   └── utils                                       # 公共工具（架构检测、坐标工具等）
│   ├── inc                                             # 公共头文件目录
│   │   ├── common                                      # 公共定义
│   │   ├── err                                         # 错误码定义
│   │   ├── fallback                                    # Fallback相关
│   │   ├── framework                                   # 框架适配头文件
│   │   ├── kernel                                      # Kernel相关头文件
│   │   ├── op_api                                      # op_api相关头文件
│   │   ├── op_graph                                    # op_graph相关头文件
│   │   ├── op_host                                     # op_host相关头文件
│   │   ├── op_kernel                                   # op_kernel相关头文件
│   │   ├── tiling_base                                 # Tiling基础头文件
│   │   ├── error_util.h                                # 错误处理工具头文件
│   │   └── op_util.h                                   # 算子工具头文件
│   ├── src                                             # 公共代码目录
│   │   ├── framework                                   # 框架适配实现
│   │   └── op_host                                     # op_host公共实现
│   └── stub                                            # Stub实现目录
│       ├── CMakeLists.txt
│       ├── inc/framework                               # 框架Stub头文件
│       └── op_api                                      # op_api Stub实现（aclnn_kernels、level0）
├── reliability                                         # RAS算子分类目录（可靠性、可用性、可维护性类算子）
│   ├── CMakeLists.txt                                  # reliability目录CMakeList入口
│   ├── ${op_name}                                      # 算子工程目录，${op_name}表示算子名（小写下划线形式），如crypto、obfuscation_calculate、obfuscation_setup
│   │   ├── CMakeLists.txt                              # 算子CMakeList入口
│   │   ├── docs                                        # 算子文档目录
│   │   │   ├── aclnn${OpName}.md                       # 算子aclnn接口介绍文档，${OpName}表示算子名（大驼峰形式）
│   │   │   └── aclnn${OpName}V2.md                     # 可选，V2版本接口文档
│   │   ├── op_host                                     # Host侧实现
│   │   │   ├── CMakeLists.txt                          # Host侧CMakeList文件
│   │   │   ├── ${op_name}_aicpu_infershape.cpp         # 可选，AI CPU场景下的InferShape实现
│   │   │   └── op_api                                  # 算子aclnn实现文件目录
│   │   │       ├── aclnn_${op_name}.cpp                # 算子aclnn接口实现文件
│   │   │       ├── aclnn_${op_name}.h                  # 算子aclnn接口实现头文件
│   │   │       ├── ${op_name}.cpp                      # 算子l0接口实现文件
│   │   │       └── ${op_name}.h                        # 算子l0接口实现头文件
│   │   └── tests                                       # 算子测试用例目录
│   │       ├── CMakeLists.txt
│   │       └── ut                                      # UT测试用例
│   │           ├── CMakeLists.txt                      # UT用例CMakeList文件
│   │           └── op_host                             # op_host测试用例目录
│   │               ├── CMakeLists.txt
│   │               ├── test_aclnn_${op_name}.cpp       # 算子aclnn测试用例文件
│   │               └── test_${op_name}_infershape.cpp  # 算子InferShape测试用例文件
│   └── ...
├── docs                                                # 项目相关文档目录
│   ├── QUICKSTART.md                                   # 快速入门指南
│   ├── README.md                                       # 文档总览
│   ├── en                                              # 英文文档目录
│   └── zh                                              # 中文文档目录
│       ├── context                                     # 基础概念与上下文文档（数据类型、数据结构、目录结构等）
│       ├── debug                                       # 调试相关文档
│       ├── develop                                     # 开发指南（AI Core、AI CPU、图开发）
│       ├── figures                                     # 文档图片资源
│       ├── invocation                                  # 算子调用说明
│       ├── op_api_list.md                              # op_api接口列表
│       └── op_list.md                                  # 算子列表
├── examples                                            # 端到端算子开发和调用示例
│   ├── add_example                                     # AI Core算子示例目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件
│   │   ├── README.md                                   # 示例介绍文档
│   │   ├── examples                                    # 算子使用示例目录
│   │   ├── op_graph                                    # 算子构图相关目录
│   │   ├── op_host                                     # 算子信息库、Tiling、InferShape相关实现目录
│   │   ├── op_kernel                                   # 算子Kernel目录
│   │   └── tests                                       # 算子测试用例目录
│   ├── add_example_aicpu                               # AI CPU算子示例目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件
│   │   ├── README.md                                   # 示例介绍文档
│   │   ├── examples                                    # 算子使用示例目录
│   │   ├── op_graph                                    # 算子构图相关目录
│   │   ├── op_host                                     # 算子信息库、InferShape相关实现
│   │   ├── op_kernel_aicpu                             # 算子Kernel目录
│   │   └── tests                                       # 算子测试用例目录
│   ├── fast_kernel_launch_example                      # 轻量级，高性能的算子开发工程模板
│   │   ├── ascend_ops                                  # 示例算子实现目录
│   │   ├── cmake                                       # 构建相关cmake脚本目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件
│   │   ├── csrc                                        # C/C++扩展源码目录
│   │   ├── README.md                                   # 轻量级，高性能的算子开发工程说明资料
│   │   ├── requirements.txt
│   │   ├── setup.py                                    # 构建脚本
│   │   └── tests                                       # 测试用例目录
│   ├── CMakeLists.txt
│   └── README.md                                       # 项目示例介绍文档
├── experimental                                        # 用户自定义算子存放目录
│   └── reliability                                     # 用户开发的reliability类算子目录
│       └── CMakeLists.txt
├── scripts                                             # 脚本目录，包含自定义算子、Kernel构建相关配置文件
│   ├── ci                                              # CI持续集成脚本
│   │   ├── gen_ops_soc.py                              # 算子SoC信息生成脚本
│   │   └── ops_run_repackage.sh                        # 算子运行重打包脚本
│   ├── custom                                          # 自定义算子安装脚本
│   │   ├── install.sh                                  # 安装脚本
│   │   └── upgrade.sh                                  # 升级脚本
│   ├── kernel                                          # Kernel构建相关脚本
│   │   ├── binary_config                               # 二进制配置文件目录（ascendc_config.json等）
│   │   └── binary_script                               # 二进制构建脚本目录（构建、打包、信息生成等）
│   ├── opgen                                           # 算子代码生成工具
│   │   ├── template                                    # 算子工程模板（add_example、add_example_aicpu）
│   │   └── opgen_standalone.py                         # 独立算子生成脚本
│   ├── package                                         # 打包相关脚本
│   │   ├── common                                      # 打包公共模块（配置、Python/Shell工具）
│   │   ├── latest_manager                              # 最新版本管理脚本
│   │   ├── module                                      # 打包模块定义
│   │   ├── ops_ras                                     # ops-ras专属打包配置
│   │   └── package.py                                  # 打包入口脚本
│   └── util                                            # 构建工具脚本目录
│       ├── ascendc_bin_param_build.py                  # Ascend C二进制参数构建
│       ├── ascendc_gen_options.py                      # Ascend C编译选项生成
│       ├── ascendc_impl_build.py                       # Ascend C实现构建
│       ├── ascendc_ops_config.py                       # Ascend C算子配置
│       ├── ascendc_replay_build.py                     # Ascend C回放构建
│       ├── build_opp_kernel_static.py                  # OPP Kernel静态库构建
│       ├── code_channel_infer.py                       # 代码通道推导
│       ├── const_var.py                                # 常量变量定义
│       ├── dependency_parser.py                        # 依赖解析
│       ├── gen_version_info.sh                         # 版本信息生成
│       ├── generate_cpp_cov.sh                         # C++覆盖率生成
│       ├── get_opfile_from_opsinfo.py                  # 从算子信息获取文件
│       ├── insert_op_info.py                           # 插入算子信息
│       ├── insert_simplified_keys.py                   # 插入简化Key
│       ├── kernel_entry.py                             # Kernel入口处理
│       ├── merge_aicpu_info_json.sh                    # 合并AI CPU信息JSON
│       ├── merge_proto.py                              # 合并Proto文件
│       ├── modify_gen_aclnn_static.py                  # 修改aclnn静态库生成
│       ├── opdesc_parser.py                            # 算子描述解析
│       ├── parse_changed_files.py                      # 解析变更文件
│       ├── parse_compile_changed_files.py              # 解析编译变更文件
│       ├── parse_ini_to_json.py                        # INI转JSON
│       ├── replay_codegen.py                           # 回放代码生成
│       └── tiling_data_def_build.py                    # TilingData定义构建
├── tests                                               # 项目级测试目录
│   ├── requirements.txt                                # 测试用例依赖的第三方组件
│   └── ut                                              # UT用例工程
│       ├── CMakeLists.txt                              # UT工程的CMakeList脚本
│       ├── empty.cpp                                   # 空文件，用于CMake编译占位
│       ├── common                                      # UT工程中使用的公共代码
│       │   ├── CMakeLists.txt
│       │   ├── any_value.h                             # 通用值类型工具
│       │   ├── infer_shape_context_faker.cpp/h         # InferShape上下文Mock
│       │   ├── infer_datatype_context_faker.cpp/h      # InferDataType上下文Mock
│       │   ├── infershape_test_util.cpp/h              # InferShape测试工具
│       │   ├── tiling_context_faker.cpp/h              # Tiling上下文Mock
│       │   ├── tiling_parse_context_faker.cpp/h        # Tiling解析上下文Mock
│       │   ├── ut_op_common.cpp/h                      # UT公共测试用例
│       │   ├── ut_op_util.cpp/h                        # UT公共工具
│       │   └── ...
│       ├── op_api                                      # op_api测试工程
│       │   ├── CMakeLists.txt
│       │   ├── op_api_ut_common                        # op_api UT公共模块
│       │   │   ├── inc                                 # 公共头文件（tensor_desc、array_desc等）
│       │   │   └── src                                 # 公共实现
│       │   ├── stub                                    # op_api测试Stub实现
│       │   └── ut_main.cpp                             # op_api测试入口
│       ├── op_host                                     # op_host测试工程
│       │   ├── CMakeLists.txt
│       │   └── test_op_host_main.cpp                   # op_host测试入口
│       ├── op_kernel                                   # op_kernel测试工程
│       │   ├── CMakeLists.txt
│       │   ├── data_utils.cpp/h                        # 数据工具
│       │   ├── scripts                                 # 测试脚本（tiling头文件生成等）
│       │   └── test_op_kernel_main.cpp                 # op_kernel测试入口
│       └── op_kernel_aicpu                             # op_kernel_aicpu测试工程
│           ├── CMakeLists.txt
│           ├── stub                                    # AI CPU测试Stub实现
│           ├── utils                                   # AI CPU测试工具（文件读取、测试辅助等）
│           └── test_op_kernel_aicpu_main.cpp           # op_kernel_aicpu测试入口
├── third_party                                         # 第三方依赖源码目录
│   ├── abseil-cpp                                      # abseil-cpp库源码
│   ├── ascend_protobuf                                 # protobuf库源码
│   └── pkg                                             # 其他第三方依赖包
├── .clang-format                                       # 代码格式化配置文件
├── CMakeLists.txt                                      # 项目工程CMakeList入口
├── CONTRIBUTING.md                                     # 项目贡献指南文件
├── LICENSE                                             # 项目开源许可证信息
├── OAT.xml                                             # 配置脚本，代码仓工具使用，用于检查License是否规范
├── README.md                                           # 项目工程总介绍文档
├── SECURITY.md                                         # 项目安全声明文件
├── Third_Party_Open_Source_Software_Notice             # 第三方开源软件声明
├── build.sh                                            # 项目工程编译脚本
├── classify_rule.yaml                                  # 组件划分信息
├── install_deps.sh                                     # 项目安装依赖包脚本
├── requirements.txt                                    # 项目的第三方依赖包
└── version.info                                        # 项目版本信息
```
