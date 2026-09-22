## Introduction

> Description:
> - Most operators in the operator library run on the AI Core, and a small number of operators run on the AI CPU. By default, the operators mentioned in this project refer to AI Core operators.
> - For the introduction to the AI Core and AI CPU, refer to "Concepts, Principles, and Terminology > Hardware Architecture and Data Processing Principles" in [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC).

This project provides development and invocation samples for AI Core operators and AI CPU operators. Refer to the corresponding implementation based on actual requirements.

## Directory Description
```
├── examples
│   ├── add_example                # AI Core operator name
│   │   ├── CMakeLists.txt         # Operator compilation configuration file. Retain the original file.
│   │   ├── examples               # Operator usage samples
│   │   ├── op_graph               # Operator graph construction directory
│   │   ├── op_host                # Operator information library, tiling, and InferShape implementation
│   │   └── op_kernel              # Operator kernel directory
│   ├── add_example_aicpu          # AI CPU operator name
│   │   ├── CMakeLists.txt         # Operator compilation configuration file. Retain the original file.
│   │   ├── examples               # Operator usage samples
│   │   ├── op_graph               # Operator graph construction directory
│   │   ├── op_host                # Operator information library and InferShape implementation
│   │   └── op_kernel_aicpu        # Operator kernel directory
│   ├── CMakeLists.txt             # Operator compilation configuration file. Retain the original file.
│   └── README.md                  # Operator description document
```

## Operator Development Samples
| Sample Directory | Sample Description | Operator Development | Operator Invocation |
|---|---|---|---|
| add_example | Operator that implements addition of two tensors. | For the end-to-end operator development process, refer to [AI Core Operator Development Guide](../docs/en/develop/aicore_develop_guide.md). | For the invocation sample, refer to [examples](./add_example/examples/). |
| add_example_aicpu | Operator that implements addition of two tensors. | For the end-to-end operator development process, refer to [AI CPU Operator Development Guide](../docs/en/develop/aicpu_develop_guide.md). | For the invocation sample, refer to [examples](./add_example_aicpu/examples/). |
| fast_kernel_launch_example | Sample that implements rapid end-to-end operator development in a PyTorch scenario. | For the end-to-end operator development process, refer to [PyTorch Operator Rapid Development Guide](./fast_kernel_launch_example/README.md). | For the invocation sample, refer to [examples](./fast_kernel_launch_example/). |
