# Documentation Center

## Directory Structure

The Docs directory structure is described as follows:

```text
├── zh
  ├── context                            # Public documents, such as terminology, basic concepts, and so on
  ├── debug                              # Operator debugging guidance documents
  │   ├── cann_sim.md
  │   ├── op_debug_prof.md
  │   └── ...
  ├── develop                            # Operator development guidance documents
  │   ├── aicore_develop_guide.md
  │   ├── aicpu_develop_guide.md
  │   ├── cross_platform_migration_guide.md
  │   ├── graph_develop_guide.md
  │   └── ...
  ├── figures                            # Image directory
  ├── install                            # Environment installation and compilation guidance documents
  │   ├── build.md
  │   ├── compile.md
  │   ├── dir_structure.md
  │   ├── quick_install.md
  │   └── ...
  ├── invocation                         # Operator invocation guidance documents (including aclnn invocation, graph mode invocation, and so on)
  │   ├── quick_op_invocation.md
  │   ├── op_invocation.md
  │   └── ...
  ├── op_api_list.md                     # Complete operator interface list (aclnn)
  ├── op_list.md                         # Complete operator list
├── CONTRIBUTING_DOCS.md                 # Documentation contribution guide
├── QUICKSTART.md                        # Quick start
└── README.md
```

## Advanced Tutorials

### Guide Documents

| Document                                                         | Description                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Source Code Build Guide](zh/install/compile.md)                        | Introduces different source code build methods and verification methods in online and offline scenarios.          |
| [Operator Invocation Guide](zh/invocation/quick_op_invocation.md)         | Introduces the method of invoking operator samples and different operator invocation methods (such as aclnn/graph, and so on).     |
| [Standard Operator Development Guide](zh/develop/aicore_develop_guide.md)       | Introduces how to define operator prototypes and implement Tiling and Kernel based on standard engineering. Such operators are called "standard operators".<br>Standard operators support aclnn and graph mode invocation. |
| [Simple Operator Development Guide](../examples/fast_kernel_launch_example/README.md) | Introduces how to implement fast_kernel_launch based on simple engineering, that is, the `<<<>>>` method. Such operators are called "simple operators".<br>Simple operators only support PyTorch invocation. |
| [Operator Debugging and Tuning](zh/debug/op_debug_prof.md)                    | Introduces common operator function debugging and performance tuning methods (such as data collection and simulation pipeline, and so on). |

### API Documents

| Document        | Description                  |
| ----------------------- | ---------------------- |
| [Operator List](zh/op_list.md)                        | Introduces the list of all operators included in the project.                                 |
| [aclnn List](zh/op_api_list.md)                   | Introduces the list of all operator aclnn APIs included in the project. To facilitate users to invoke operators on the Host side, C language APIs are provided, that is, APIs with the aclnn prefix. |

### Tool Documents

| Document        | Description                  |
| ----------------------- | ---------------------- |
| [Simulator Tool](zh/debug/cann_sim.md) | A SoC-level simulation tool for operator development scenarios, used to analyze the precision and performance data of AI tasks running on the AI simulator at each stage. |

### More Documents

- Sample Documents: Refer to the operator samples in the [cann-samples](https://gitcode.com/cann/cann-samples) repository.

## Appendix

| Document                                | Description                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [Operator Basic Concepts](zh/context/basic_concept.md) | Introduces basic concepts and terminology in the operator domain, such as quantization/sparse, data types, data formats, and so on. |
| [build Parameter Description](zh/install/build.md)   | Introduces the functions and parameter values of the build.sh script in this project, including source code compilation, operator invocation, debugging, and so on. |