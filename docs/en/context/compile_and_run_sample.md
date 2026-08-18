# Compilation and Running Examples

## Prerequisites

- If you need to compile and execute operator APIs, ensure that the basic environment has been set up, including driver, firmware, CANN software package, ops package, etc.
- For the operator API calling process and compilation and running operations, refer to [Application Development (C&C++)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/programug/acldevg/aclcppdevg_000006.html) under "Single Operator Invocation > Single Operator API Execution > Calling aclnn Interface Example Code".

## Pre-compilation Preparation

This chapter uses a co-located development and runtime environment, where the machine with an AI processor is used for both development and execution. The **AddExample operator** is used below. Other operators follow the same general calling and compilation flow; adapt the API source file (*.cpp) and CMakeLists.txt as needed.

- **Example Code**

   AddExample performs element-wise tensor addition. Use [test_aclnn_add_example.cpp](../../../examples/add_example/examples/test_aclnn_add_example.cpp) as the example source file and name it `test_aclnn_add_example.cpp`.

- **CMakeLists File**

    The CMake file example is as follows. Please modify according to the actual situation:

    ```bash
    # Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.14)

    # Set project name
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # Set compilation options
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # Set executable file name (such as opapi_test) and specify the directory where the operator file *.cpp to be run is located
    add_executable(opapi_test
                   test_aclnn_add_example.cpp)

    # Set ASCEND_PATH (CANN software package directory, please modify according to the actual path) and INCLUDE_BASE_DIR (header file directory)
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

    # Set linked library file path
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${ASCEND_PATH}/lib64/librasopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_ras.so)

    # The executable file is in the bin directory under the CMakeLists file directory
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

## Compilation and Running

  1. Prepare the operator calling code (*.cpp) and compilation script (CMakeLists.txt) in advance.
  2. Configure environment variables.

     After installing the CANN software, log in to the environment as the CANN runtime user and execute the following command to make the environment variables effective.

        ```bash
        source ${INSTALL_DIR}/set_env.sh
        ```

     Where ${INSTALL_DIR} is the storage path after CANN software installation. Please replace according to the actual situation.
  3. Compile and run.
        - Enter the directory where CMakeLists.txt is located and execute the following command to create a new build directory to store the generated compilation files.

            ```bash
            mkdir -p build
            ```

        - Enter the build directory, execute the cmake command to compile, and then execute the make command to generate the executable file.

          ```bash
          cd build
          cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
          make
          ```

          After successful compilation, the opapi\_test executable file will be generated in the bin folder under the build directory.

        - Enter the bin directory and run the executable file opapi_test.

          ```bash
          cd bin
          ./opapi_test
          ```

          Taking the running result of AddExample as an example, the output is similar to the following:

          ```bash
          result[0] is: 1.200000
          result[1] is: 2.200000
          result[2] is: 3.200000
          result[3] is: 5.400000
          result[4] is: 6.400000
          result[5] is: 7.400000
          result[6] is: 9.600000
          result[7] is: 10.600000
          ```

          If the execution result reports an error and the expected result does not appear, you can use the aclGetRecentErrMsg interface to obtain the specific error information.
          Example of obtaining exception information when calling `aclnnAddExampleGetWorkspaceSize` fails:

          ```bash
          // selfX is nullptr
          ret = aclnnAddExampleGetWorkspaceSize(selfX, selfY, out, &workspaceSize, &executor);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExampleGetWorkspaceSize failed. ERROR: %d\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
          ```

          The above null pointer construction problem obtains error information as shown below:

          ```bash
          aclnnAddExampleGetWorkspaceSize failed. ERROR: 161001
          [ERROR msg][PID:xxxx] xxx(timestamp) AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for an input argument.
          ```
