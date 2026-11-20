# Installation Guide

<!-- md-trans-meta sourceCommit=c020bdb62acb53e4224a27aecdcce6019c4e8cd5 translatedAt=2026-11-11T08:58:34.403Z pushedAt=2026-11-12T01:07:09.484Z -->

This document provides the source code compilation and installation method for Kunpeng ONNX Runtime, organized in the order of preparing the environment, obtaining dependencies, compiling, and verifying, and is intended for O&M and build personnel responsible for environment setup.

## Verified Environments

To ensure that you can use the features of Kunpeng ONNX Runtime smoothly and securely, ensure that the environment used is within the verified environment scope.

**Hardware Requirements**

The verified hardware environment is shown in [**Table 1** Hardware requirements](#hardware-requirements).

**Table 1** Hardware requirements<a id="hardware-requirements"></a>

| Item | Description |
| ---- | ---------- |
| CPU  | Kunpeng 950 processor |

**OS Requirements**

The verified OSs are shown in [**Table 2** OS requirements](#os-requirements).

**Table 2** OS requirements<a id="os-requirements"></a>

| Item | Version | Description | Download |
| ---- | ---- | ---- | ---- |
| OS   | openEuler 24.03 LTS SP3 | If a fresh OS installation is performed, you can choose the "Minimal Install" mode and select the Development Tools suite. Otherwise, many software packages need to be manually installed. | [openEuler download link](https://repo.openeuler.org/openEuler-24.03-LTS-SP3/ISO/aarch64/) |

**Software Requirements**

The verified software dependency environment is shown in [**Table 3** Software requirements](#software-requirements).

**Table 3** Software requirements<a id="software-requirements"></a>

| Item | Version | Description | Download |
| ---- | ---- | ---- | ---- |
| Python | 3.13.18 | Python is an auxiliary tool in the ONNX Runtime build process; it automatically downloads dependencies and configures the environment. | Install from the Yum source. |
| CMake | 3.30.3  | CMake is the build tool for ONNX Runtime, and CMake version 3.28.3 or later is required. | Install from the Yum source. |
| GCC/G++ | 12.3.1  | GCC (GNU Compiler Collection) is a programming language compiler. | Install from the Yum source. |

## Preparing Feature Dependencies

### KDNN

#### Downloading the KDNN Package and Verification File

Visit the [KDNN release page](https://gitcode.com/boostkit/boostsra/releases), locate suite release v1.5.0, and download the following two files:

- KDNN package: `BoostKit-boostcore-kdnn_3.3.0.zip`
- SHA256 verification file: `BoostKit-boostcore-kdnn_3.3.0.zip.sha256`

#### Verifying Package Integrity

To check whether a package is incomplete due to network or device problems during transfer or storage, verify package integrity after obtaining it. Only packages that pass the verification can be deployed.<br>
Integrity is verified by comparing the checksum recorded in the verification file with the package checksum calculated manually. If the two values are the same, the file is complete. Otherwise, the file integrity is compromised. Obtain the package again.<br>
Before verifying package integrity, prepare the following files.

> ![icon note](public_sys-resources/icon-note.gif) **NOTE**
> In the following package `BoostKit-boostcore-kdnn_xxx.zip`, replace the placeholder `xxx` with the actual version.

- Package: `BoostKit-boostcore-kdnn_xxx.zip`
- Verification file: .sha256 file with the same name

The procedure for verifying file integrity is as follows:

1. Calculate the SHA256 checksum of the package.

   - On Linux, run:

        ```bash
      sha256sum BoostKit-boostcore-kdnn_xxx.zip
      ```

   - On Windows, run:

        ```bash
       certutil -hashfile BoostKit-boostcore-kdnn_xxx.zip SHA256
       ```

      After the command is executed, the checksum is output.

2. Compare the checksum calculated in step 1 with the SHA256 checksum in the verification file.

   If the checksums are the same, the file is complete. If the checksums are different, the file integrity has been compromised and the file needs to be obtained again.

#### Installing the KDNN Package

- Decompress the KDNN package and install it using the `rpm -ivh` command.

  ```bash
  unzip BoostKit-boostcore-kdnn_3.3.0.zip
  rpm -ivh boostcore-kdnn-3.3.0-1.aarch64.rpm
  ```

  After installation, the header file is located in `/usr/local/kdnn/include`, and the static library is located in `/usr/local/kdnn/lib`.

## Compiling and Installing ONNX Runtime

1. Obtain the open-source ONNX Runtime code. Assume that the installation path is `/path/to`.

   ```bash
   cd /path/to
   git clone --branch v1.26.0 https://github.com/microsoft/onnxruntime.git
   ```

2. Download and apply the patches.

   ```bash
   git clone --branch master https://gitcode.com/boostkit/onnxruntime.git kp_onnxruntime
   cd onnxruntime
   git apply ../kp_onnxruntime/patches/001-boostsra-onnxruntime-1.0.0-graph-optimization.patch
   git apply ../kp_onnxruntime/patches/002-boostsra-onnxruntime-1.0.0-parallel-executor.patch
   ```

3. Place the KDNN header file and static library into the generated ONNX Runtime source code.

   ```bash
   export ORT_PATH=/path/to/onnxruntime
   mkdir -p $ORT_PATH/onnxruntime/core/kdnn/out
   cp -r /usr/local/kdnn/include $ORT_PATH/onnxruntime/core/kdnn/out
   cp -r /usr/local/kdnn/lib $ORT_PATH/onnxruntime/core/kdnn/out
   ```

4. Build and install ONNX Runtime.

   ```bash
   # Build ONNX Runtime.
   ./build.sh --config Release --build_shared_lib --parallel \
     --compile_no_warning_as_error --skip_submodule_sync \
     --allow_running_as_root --skip_tests --build_wheel \
     --cmake_extra_defines onnxruntime_USE_KDNN=ON onnxruntime_USE_SVE=ON \
     --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_Protobuf=ON \
     --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
   # Install ONNX Runtime. The following installation package is used as an example for Python 3.13.
   pip install build/Linux/Release/dist/onnxruntime-1.26.0-cp313-cp313-linux_aarch64.whl
   ```

   Where:

   - `onnxruntime_USE_KDNN=ON`: Builds and registers the internal KDNN operators and MHA graph fusion.
   - `onnxruntime_USE_SVE=ON`: Enables the SVE build configuration required by ORT and the target platform.
   - contrib ops must remain enabled. Do not use `onnxruntime_DISABLE_CONTRIB_OPS=ON`.

## Building and Installing ANNC

1. Install the ANNC prerequisites.

   ```bash
   yum install cmake ninja-build protobuf-devel protobuf-compiler \
         gcc gcc-c++ python3-devel numactl \
         llvm-toolset-20-llvm llvm-toolset-20-llvm-devel \
         llvm-toolset-20-llvm-libs llvm-toolset-20-llvm-static \
         llvm-toolset-20-llvm-mlir llvm-toolset-20-llvm-mlir-devel \
         llvm-toolset-20-llvm-mlir-static llvm-toolset-20-python3-mlir \
         llvm-toolset-20-clang
   pip install numpy
   ```

2. Obtain the ANNC source code.

   ```bash
   cd /path/to
   git clone --branch dev https://gitcode.com/openeuler/ANNC.git
   ```

3. Generate the ONNX protobuf.

   ```bash
   cd /path/to/ANNC
   export ANNC_PATH=/path/to/ANNC
   ./frontends/onnx/protos/onnx_protos_minimal.sh -d $ANNC_PATH
   ```

4. Configure and build the ORT acceleration stack.

   ```bash
   export ORT_PATH=/path/to/onnxruntime
   export ANNC_ONNXRUNTIME_INCLUDE_DIR=$ORT_PATH/include/onnxruntime/core/session
   export ANNC_EP_ORT_BUILD_TREE=$ORT_PATH/build/Release
   export ANNC_MLAS_INCLUDE_DIR=$ORT_PATH/onnxruntime/core/mlas/inc

   ./build.sh --no-install-deps \
      --build-type Release --install-prefix $ANNC_PATH/install --enable-ort
   ```

5. Build and install ANNC.

   ```bash
   export ANNC_PATH=/path/to/ANNC
   ninja -C $ANNC_PATH/build -j16
   ninja -C $ANNC_PATH/build install
   ```

   `j16` indicates the number of CPU cores used for compilation. Users can set this value based on their container's CPU quota when compiling.
   After ANNC is successfully installed, the following key artifacts can be found in the `/path/to/ANNC/install` path.

   ```bash
   /bin/annc-onnx-pipeline
   /bin/annc-onnx2atir
   /bin/annc
   /bin/annc-opt
   /bin/annc-asm
   /lib64/libannc_ep.so
   /lib64/libANNC*.a
   ```

## Verifying the Installation

After the installation is complete, run the following Python script to verify whether ONNX Runtime, the KDNN library, and the ANNC library are successfully installed.

```python
import onnxruntime as ort
import os
import ctypes
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

ctypes.CDLL(
    os.path.join(os.path.dirname(ort.__file__), "capi", "libonnxruntime.so.1.26.0"),
    mode=ctypes.RTLD_GLOBAL)
# Replace the path below with the actual /path/to/ANNC/install/lib64/libannc_ep.so path
rtpy.register_execution_provider_library(
    "AnncExecutionProvider", "/path/to/ANNC/install/lib64/libannc_ep.so")  

print(ort.__version__)    # Expected output: ONNX Runtime version 1.26.0
print(ort.get_available_providers())   # Expected output: ['KDNNExecutionProvider', 'CPUExecutionProvider', 'AnncExecutionProvider']
```

After the preceding script is executed, if the following content is output, ONNX Runtime, the KDNN library, and the ANNC library are successfully installed.

```output
1.26.0
['KDNNExecutionProvider', 'CPUExecutionProvider', 'AnncExecutionProvider']
```

## Change History

| Version | Date | Description |
| ---- | ---- | ---- |
| 01 | 2026-12-30 | This is the first official release. |
