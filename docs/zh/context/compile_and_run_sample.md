# 编译与运行样例

## 前提说明

- 如需编译执行算子API，请确保基础环境已搭建完成，包括驱动、固件、CANN软件包、ops包等。
- 算子API的调用流程和编译运行操作详情请参见[《应用开发（C&C++）》](https://hiascend.com/document/redirect/CannCommunityCppInferWizard)中“单算子调用>单算子API执行>调用aclnn接口示例代码”。

## 编译前准备

本章以开发和运行环境合设场景为例，即带AI处理器的机器既作为开发环境又作为运行环境。该场景下，代码开发和代码运行在同一台机器上。这里以**AddExample算子**为例，其他算子的调用逻辑、流程、编译脚本与AddExample算子大致一样，请根据实际情况自行修改API调用脚本（\*.cpp）和编译脚本(CMakeLists)。

- **示例代码**

   已知AddExample算子实现了张量加法运算。您可以从[test_aclnn_add_example.cpp](../../../examples/add_example/examples/test_aclnn_add_example.cpp)中获取示例代码，并将代码文件命名为“**test\_aclnn\_add\_example.cpp**”。

- **CMakeLists文件**

    CMake文件示例如下，请根据实际情况修改：

    ```cpp
    # Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.14)

    # 设置工程名
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # 设置编译选项
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # 设置可执行文件名（如opapi_test），并指定待运行算子文件*.cpp所在目录
    add_executable(opapi_test
                   test_aclnn_add_example.cpp)

    # 设置ASCEND_PATH（CANN软件包目录，请根据实际路径修改）和INCLUDE_BASE_DIR（头文件目录）
    if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
        set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
        ${INCLUDE_BASE_DIR}
        ${INCLUDE_BASE_DIR}/aclnn
    )

    # 设置链接的库文件路径
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${ASCEND_PATH}/lib64/librasopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_ras.so)

    # 可执行文件在CMakeLists文件所在目录的bin目录下
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

## 编译与运行

  1. 提前准备好算子的调用代码（\*.cpp）和编译脚本（CMakeLists.txt）。
  2. 配置环境变量。

     安装CANN软件后，使用CANN运行用户登录环境，执行如下命令生效环境变量。

        ```bash
        source ${INSTALL_DIR}/set_env.sh
        ```

     其中${INSTALL_DIR}为CANN软件安装后文件存储路径，请根据实际情况替换。
  3. 编译并运行。
        - 进入CMakeLists.txt所在目录，执行如下命令，新建build目录存放生成的编译文件。

            ```bash
            mkdir -p build
            ```

        - 进入build目录，执行cmake命令编译，再执行make命令生成可执行文件。

          ```bash
          cd build
          cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
          make
          ```

          编译成功后，会在build目录的bin文件夹下生成opapi\_test可执行文件。

        - 进入bin目录，运行可执行文件opapi_test。

          ```bash
          cd bin
          ./opapi_test
          ```

        以AddExample算子的运行结果为例，运行后的结果示例如下：

        ```text
        result[0] is: 1.200000
        result[1] is: 2.200000
        result[2] is: 3.200000
        result[3] is: 5.400000
        result[4] is: 6.400000
        result[5] is: 7.400000
        result[6] is: 9.600000
        result[7] is: 10.600000
        ```
