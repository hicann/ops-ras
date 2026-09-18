# Contribution Guide

This project welcomes developers to experience and participate in contributions. Before participating in community contributions, please see [cann-community](https://gitcode.com/cann/community) to understand the code of conduct, sign the CLA agreement, and understand the contribution process of the source code repository.

Developers need to pay attention to the following points when preparing local code and submitting PRs:

1. When submitting a PR, please carefully fill in the business background, purpose, solution, and other information of this PR according to the PR template.
2. If your modification is not a simple bug fix, but involves adding new features, new interfaces, new configuration parameters, or modifying code flow, please be sure to discuss the solution through an Issue first to avoid your code being rejected. If you are not sure whether this modification can be classified as a "simple bug fix", you can also discuss the solution by submitting an Issue.

Developer contribution scenarios mainly include:

- Operator Bug Fix

  If you discover certain operator bugs in this project and want to fix them, we welcome you to create a new Issue for feedback and tracking.

  You can create a new `Bug-Report|Bug Report` type Issue according to the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to describe the bug, and then enter "/assign" or "/assign @yourself" in the comment box to assign this Issue to you for processing.

- Operator Optimization

  If you have generalization enhancement/performance optimization ideas for certain operator implementations in this project and want to implement these optimization points, we welcome you to contribute operator optimizations.

  You can create a new `Requirement|Feature Request` type Issue according to the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to explain the optimization points and provide your design solution, and then enter "/assign" or "/assign @yourself" in the comment box to assign this Issue to you for tracking optimization.

- Contribute New Operators

  If you have a brand new operator that you want to design and implement based on NPU, we welcome you to propose new ideas and designs in an Issue.

  You can create a new `Requirement|Feature Request` type Issue according to the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to provide the new operator description and design solution. Project members will communicate and confirm with you, and provide a suitable `contrib` directory classification for your operator under the `experimental` directory. You can contribute the new operator to the corresponding directory.

  At the same time, you need to comment "/assign" or "/assign @yourself" in the submitted Issue to claim this Issue and subsequently complete the new operator submission.

  The deliverables for new operators are usually quite numerous. You can refer to the following list to check the minimum deliverable set, where `${op_name}` indicates the new operator name.
  ```
  ${op_class}                                          # operator classification
  ├── ${op_name}                                       # operator name
  │   ├── op_host                                      # operator definition, Tiling, InferShape related implementation
  │   │   ├── ${op_name}_def.cpp                       # operator definition file
  │   │   ├── ${op_name}_tiling.cpp                    # operator Tiling implementation file
  │   │   ├── ${op_name}_tiling_${sub_case}.cpp        # Optional, sub-scenario Tiling implementation. ${sub_case} represents the sub-scenario (e.g., arch35)
  │   │   ├── ${op_name}_tiling_${sub_case}.h          # Optional, sub-scenario Tiling implementation header file
  │   │   └── CMakeLists.txt
  │   ├── op_kernel                                    # operator Kernel directory
  │   │   ├── ${op_name}.cpp
  │   │   ├── ${op_name}.h
  │   │   ├── ${op_name}_tiling_data.h
  │   │   ├── ${op_name}_tiling_key.h
  │   │   └── CMakeLists.txt
  │   ├── CMakeLists.txt                               # operator compilation configuration file, keep the original file
  │   └── README.md                                    # operator description document
  ```

  For the complete operator directory structure (including optional deliverables), see [Project Directory](docs/en/install/dir_structure.md#project-directory).

  > **Note**: Tiling implementation files under the op_host directory that participate in compilation must contain the `_tiling` identifier in the file name (e.g., `${op_name}_tiling.cpp`, `${op_name}_tiling_${sub_case}.cpp`); otherwise, they will not be recognized by the compilation system. When splitting Tiling implementations for sub-scenarios (e.g., a specific architecture such as arch35), follow this naming rule.

- Document Correction

  If you discover certain operator document description errors in this project, we welcome you to create a new Issue for feedback and correction.

  You can create a new `Documentation|Documentation Feedback` type Issue according to the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to point out the problems in the corresponding document, and then enter "/assign" or "/assign @yourself" in the comment box to assign this Issue to you to correct the corresponding document description.

- Help Solve Others' Issues

  If you have suitable solutions for problems encountered by others in the community, we welcome you to comment and communicate in the Issue to help others solve problems and pain points, and jointly optimize usability.

  If the corresponding Issue requires code modification, you can enter "/assign" or "/assign @yourself" in the Issue comment box to assign this Issue to you for tracking and assisting in solving the problem.