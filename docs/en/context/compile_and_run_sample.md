# Compile and Run Samples

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-30T01:45:37.819Z pushedAt=2026-07-30T03:35:18.393Z -->

## Prerequisites

- To compile and run an operator API, ensure that the basic environment is set up, including the driver, firmware, CANN software package, and ops package.

- For details about the operator API calling process and compilation and running operations, see "Single Operator Calling > Single Operator API Execution > Sample Code for Calling aclnn APIs" in [*Application Development (C & C++)*](https://hiascend.com/document/redirect/CannCommunityCppInferWizard).

## Preparations Before Compilation

This chapter uses the combined development and runtime environment as an example, where the machine with an AI processor serves as both the development environment and the runtime environment. In this scenario, code development and execution are performed on the same machine. The **AddMatMul operator** is used as an example here. The call logic, process, and build script of other operators are largely the same as those of the AddMatMul operator. Modify the API call script (\*.cpp) and build script (CMakeLists) based on the actual situation.

- **Sample Code**

   The AddMatMul operator implements tensor addition, with the formula: out = β  self + α  (mat1 @ mat2). You can obtain the sample code from the "Example" section in [aclnnAddmm&aclnnInplaceAddmm](https://gitcode.com/cann/ops-nn/blob/9.0.0/matmul/mat_mul_v3/docs/aclnnAddmm&aclnnInplaceAddmm.md) and name the code file "**test_addmm.cpp**".

- **CMakeLists**

The following is an example of the CMake file. Modify it based on the actual situation:

    ```
    # Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.14)

    # Set the project name.
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # Set the compilation options.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # Set the executable file name (for example, opapi_test) and specify the directory of the operator file *.cpp to be run.
    add_executable(opapi_test
                   test_addmm.cpp)

    # Set ASCEND_PATH (the CANN package directory; modify it based on the actual path) and INCLUDE_BASE_DIR (the header file directory).
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

    # Set the linked library file path.
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${ASCEND_PATH}/lib64/librasopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_math.so
                          ${ASCEND_PATH}/lib64/libopapi_ras.so)

    # The executable file is located in the bin directory under the CMakeLists file directory.
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

    Operators that fuse and parallelize collective communication and MatMul are collectively referred to as MC2 operators, including AllGatherMatmul, AlltoAllAllGatherBatchMatMul, BatchMatMulReduceScatterAlltoAll, MatmulAllReduce, MatmulAllReduceAddRmsNorm, and MatmulReduceScatter. When calling these operator APIs, multi-threading and HCCL (Huawei Collective Communication Library) are generally involved. Therefore, the CMake file needs to additionally import the following content; otherwise, compilation will fail.

  ```
  # Set the linked library file path.
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/lib64/libascendcl.so
                        ${ASCEND_PATH}/lib64/librasopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so
                        ${ASCEND_PATH}/lib64/libopapi_ras.so
                        ${ASCEND_PATH}/lib64/libhccl.so      # Collective communication library file
                        ${CMAKE_THREAD_LIBS_INIT})           # Multithreading-dependent library files
  ```

"find_package(Threads REQUIRED)" is a CMake command used to locate the thread library, which automatically links the header files or indirectly dependent library files required by the thread library.

## Compile and Run

  1. Prepare the operator call code (\*.cpp) and the build script (CMakeLists.txt) in advance.

  2. Configure environment variables.

     After installing the CANN software, log in to the environment as the CANN running user and run the following command to make the environment variables take effect.

        ```
        source ${INSTALL_DIR}/set_env.sh
        ```

     In the command, ${INSTALL_DIR} is the path where the CANN software is installed. Replace it with the actual path.

  3. Compile and run.

        - Go to the directory where CMakeLists.txt is located and run the following command to create a build directory for storing the generated compilation files.

            ```
            mkdir -p build
            ```

        - Go to the build directory, run the cmake command for compilation, and then run the make command to generate the executable file.

          ```
          cd build
          cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
          make
          ```

          After successful compilation, the opapi\_test executable file is generated in the bin folder under the build directory.

      - Go to the bin directory and run the opapi_test executable file.

        ```bash
        cd bin
        ./opapi_test
        ```

        Taking the AddMatMul operator as an example, the result after running is as follows:

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
