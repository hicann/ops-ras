# Environment Deployment

Before performing the operations in the [Learning Tutorials](../../../README_en.md#learning-tutorials), please complete the basic environment setup and source code download by following the steps below, and ensure that the NPU driver, firmware, and CANN software (`Ascend-cann-toolkit` and `Ascend-cann-ops`) have been installed.

## Environment Installation

This project provides multiple ways to set up the Ascend environment. Choose as needed.

> **Note**: The compilation and runtime scenarios mentioned in this document are defined as follows. Choose based on your actual situation.
>
> - Compilation: For scenarios where only this project is compiled without running, you only need to install the CANN toolkit package.
> - Runtime: For scenarios where this project is run (compilation and running, or pure running), you need to install the driver and firmware, the CANN toolkit package, and the CANN ops package.

|  Installation Method  |  Description  |  Applicable Scenario  |
| ----- | ------ | ------ |
|  CANNLab  | One-stop development platform that provides an online Ascend environment that can be run directly, without manual installation.<br>Currently, single-node computing power is available, and the **latest CANN package is installed by default**. | Suitable for developers without Ascend devices.|
|  Docker  | Docker images are an efficient deployment method with CANN packages and required dependencies pre-integrated.<br>Currently applicable to Atlas A2 and A3 series products. Supported OSs: ubuntu22.04 and openeuler24.03. The **latest CANN package is installed by default**. |Suitable for developers who have Ascend devices and need to quickly set up an environment.|
|  Manual Installation  | Manually install CANN packages and basic dependencies, with high flexibility. |Suitable for developers who have Ascend devices and want to manually install CANN packages or experience the latest master branch capabilities.|

### Method 1: CANNLab

For developers without Ascend devices, you can directly use the CANNLab cloud development environment, that is, the "**one-stop development platform**". This platform provides an online Ascend environment that can be run directly. The environment has the required drivers, firmware, software packages, and dependencies installed, so manual installation is not required.

> **Note**: The environment installs the latest CANN package by default. When downloading the source code, ensure that it matches the software version. For more information about the development platform, refer to the [CANNLab Guide](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md).

1. Enter the open source project and click the "`CANNLab`" button. Log in with a certified Huawei Cloud account. If you have not registered or certified, please register and certify as prompted.

   <img src="../figures/cloudIDE.png" alt="Cloud Platform"  width="750px" height="85px">

2. Create an NPU environment and configure specifications as prompted. After the cloud development environment is started, click "`Connect > WebIDE`" to enter the one-stop development platform. By default, the resources of this open source project are stored in the `/mnt/workspace/gitCode` directory.

   <img src="../figures/webIDE.png" alt="Cloud Platform"  width="1000px" height="150px">

### Method 2: Docker Deployment

For developers with Ascend devices who want to quickly set up an Ascend environment, Docker image deployment is recommended.

> **Note**:
>
> - The image file is relatively large and takes some time to download. Please wait patiently. For options of docker commands, run `docker --help`.
> - The environment installs the latest CANN package by default. When downloading the source code, ensure that it matches the software version.

1. **Install the driver (runtime dependency)**

    The driver is a runtime dependency and can be skipped if you only compile operators. Run `npu-smi info` to check whether NPU information is displayed. If no, install the driver by referring to [CANN Quick Installation](https://www.hiascend.com/en/cann/download):

    - Step 1: On the page, select your product series, CPU architecture, and operating system, and select **Online Installation (Yum)** as the installation mode.
    - Step 2: Follow the instructions on the page to complete the three procedures: **Configuring the user group**, **Installing dependencies and configuring the source**, and **Installing the NPU driver**.
    - Step 3: Run `npu-smi info`. If NPU device information is properly displayed, the driver is successfully installed.

2. **Download the image**

    - Step 1: Log in to the host as the root user. Ensure that Docker Engine (v1.11.2 or later) has been installed on the host. Run `docker --version` to check the Docker version. If Docker is not installed, refer to the [Docker official installation guide](https://docs.docker.com/engine/install/).
    - Step 2: Pull the image with the CANN software packages and dependencies required for operator development pre-integrated from the [Ascend image repository](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884).

    The following is an example. Replace the CANN version, chip series, operating system, and Python version as required. For supported values of each field, see the Ascend image repository page.

    ```bash
    # Use cann:9.1.0-beta.1 as an example
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12-devel
    ```

    > **Note**: The image tag format is `<CANN version>-<chip series>-<OS>-<Python version>-devel`. Images with the `-devel` suffix are operator development images, which contain the dependencies for operator development and compilation.

3. **Run Docker**

    After pulling the image, start the container with specific parameters so that the container can access the Ascend devices on the host.

    ```bash
    docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12-devel bash
    ```

    > **Note**: `--name` specifies the container name. Replace it with a recognizable custom name. If the name is already in use, the container fails to start.

    | Parameter | Description | Notes |
    | :--- | :--- | :--- |
    | `--name cann_container` | Specifies a name for the container for easy management. | Can be customized. |
    | `--device /dev/davinci0` | Key configuration: maps the NPU device card of the host to the container. Multiple NPU device cards can be mapped. | Must be adjusted based on the actual situation: `davinci0` corresponds to the 0th NPU card in the system. Run the `npu-smi info` command on the host first, and modify the number based on the device number displayed in the output (such as `NPU 0` and `NPU 1`).|
    | `--device /dev/davinci_manager` | Maps the NPU device management interface. | - |
    | `--device /dev/devmm_svm` | Maps the device memory management interface. | - |
    | `--device /dev/hisi_hdc` | Maps the communication interface between the host and the device. | - |
    | `-v /usr/local/dcmi:/usr/local/dcmi` | Mounts the tools and libraries related to the device container management interface (DCMI). | - |
    | `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | Mounts the `npu-smi` tool. | Enables you to run this command in the container to query NPU status and performance information.|
    | `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | Key mount: maps the NPU driver library of the host to the container. | - |
    | `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | Mounts the driver version information file. | - |
    | `-v /etc/ascend_install.info:/etc/ascend_install.info` | Mounts the CANN software installation information file. | - |
    | `-it` | Combination of `-i` (interactive) and `-t` (allocating a pseudo terminal). | - |
    | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12-devel` | Specifies the Docker image to run. | Ensure that the image name and tag are exactly the same as those of the image you pulled using `docker pull`. |
    | `bash` | Command executed immediately after the container starts. | - |

### Method 3: Manual Installation

For developers with Ascend devices who want to manually set up an Ascend environment, refer to the following steps.

#### Installing Software

- **Scenario 1: Experiencing the latest master capabilities or developing based on the CANN weekly version**

    Refer to [CANN Quick Installation](https://www.hiascend.com/en/cann/download), select `Weekly version`, download the corresponding package based on the product series, CPU architecture, and operating system, and complete the installation by running the commands provided on the page.

- **Scenario 2: Experiencing branch capabilities or developing based on the CANN stable version**

    Refer to [CANN Quick Installation](https://www.hiascend.com/en/cann/download), select `Stable version` (only CANN 8.5.0 and later versions are supported), download the corresponding package based on the product series, CPU architecture, and operating system, and complete the installation by running the commands provided on the page.

#### Installing Basic Dependencies

The basic dependencies of this project are as follows. Ensure that the version requirements are met.

- python >= 3.7.0 (recommended version <= 3.10)
- gcc >= 7.3.0
- cmake >= 3.16.0
- pigz (optional, installing it can improve packaging speed, recommended version >= 2.4)
- dos2unix
- gawk
- make
- patch
- googletest (required only when executing UT, recommended version [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0))

The preceding dependencies can be installed at once using the project script. The procedure is as follows:

1. Download the source code.

    Download the branch source code matching the CANN version. The command is as follows. Replace `${tag_version}` with the branch tag name.

    ```bash
    git clone -b ${tag_version} https://gitcode.com/cann/ops-ras.git
    ```

2. Install the dependencies.

    First, run `install_deps.sh` in the project root directory to install the preceding dependencies at once. The command is as follows. If your system is not supported, refer to the file and adapt it yourself.

    ```bash
    bash install_deps.sh
    ```

    Then, run `requirements.txt` in the project root directory to install the third-party Python library dependencies. The command is as follows.

    ```bash
    pip3 install -r requirements.txt
    ```

## Environment Verification

After the CANN packages are installed, verify that the environment and driver are normal.

- **Check the NPU device**

    ```bash
    # Run npu-smi. If device information is properly displayed, the driver is normal
    npu-smi info
    ```

- **Check the CANN version**

    ```bash
    # View the CANN toolkit package version information (default installation path)
    # Docker and manual installation scenarios:
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info
    # CANNLab scenario:
    cat /home/developer/Ascend/cann/${arch}-linux/ascend_toolkit_install.info

    # View the CANN ops package version information (default installation path)
    # Docker and manual installation scenarios:
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_ops_install.info
    # CANNLab scenario:
    cat /home/developer/Ascend/cann/${arch}-linux/ascend_ops_install.info
    ```

    `${arch}` indicates the current architecture, which can be queried using `uname -m`, for example, aarch64 or x86_64.

## Environment Variable Configuration

Select the appropriate command as needed to make the environment variables take effect.

```bash
# Default installation path, taking the root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Specified installation path
# source ${install_path}/cann/set_env.sh
```
