# 安装指南

本文档提供鲲鹏ONNX Runtime的源码编译安装方法，整体按准备环境—获取依赖—编译—验证的顺序组织，面向负责环境搭建的运维与构建人员。

## 已验证环境

为保证您可以顺利安全地使用鲲鹏ONNX Runtime特性，请确保所使用的环境信息在已验证环境范围内。

**硬件要求**

已验证的硬件环境如[**表 1** 硬件要求](#硬件要求)所示。

**表 1** 硬件要求<a id="硬件要求"></a>

| 项目 | 说明                |
| ---- | ------------------- |
| CPU  | 鲲鹏950 7592C处理器 |

**操作系统要求**

已验证的操作系统要求如[**表 2** 操作系统要求](#操作系统要求)所示。

**表 2** 操作系统<a id="操作系统要求"></a>

| 项目 | 版本 | 说明 | 下载地址 |
| ---- | ---- | ---- | ---- |
| OS   | openEuler 24.03 LTS SP3 | 如果是全新安装操作系统，可选择“Minimal Install”安装方式并勾选Development Tools套件，否则很多软件包需要手动安装。 | [openEuler操作系统下载链接](https://repo.openeuler.org/openEuler-24.03-LTS-SP3/ISO/aarch64/) |

**软件要求**

已验证的软件依赖环境如[**表 3** 软件要求](#软件要求)所示。

**表 3** 软件要求<a id="软件要求"></a>

| 项目 | 版本 | 说明 | 下载地址 |
| ---- | ---- | ---- | ---- |
| Python  | 3.13.18 | Python是ONNX Runtime的构建过程中的辅助工具，起到自动下载依赖、配置环境等作用。 | 通过Yum源方式安装。 |
| CMake   | 3.30.3  | CMake是ONNX Runtime的构建工具，要求CMake版本为3.28.3及以上。 | 通过Yum源方式安装。 |
| GCC/G++ | 12.3.1  | GCC（GNU Compiler Collection）是一种编程语言编译器。 | 通过Yum源方式安装。 |

## 准备特性依赖

### KDNN

#### 下载KDNN软件包及校验文件

访问[KDNN 发布网页](https://gitcode.com/boostkit/boostsra/releases)，找到套件发行版v1.5.0，下载下面两个文件:

- KDNN软件包: `BoostKit-boostcore-kdnn_3.3.0.zip`
- SHA256校验文件: `BoostKit-boostcore-kdnn_3.3.0.zip.sha256`

#### 软件包完整性校验

为了检查软件包在传输或存储过程中是否因网络或设备问题而不完整，在获取到软件包后，需要对软件包的完整性进行校验，通过了校验的软件包才能部署。<br>
这里通过对比校验文件中记录的校验值和手动方式计算的软件包校验值，判断软件包是否完整。若两个值相同，说明文件完整，否则，文件完整性被破坏，请重新获取软件包。<br>
在校验软件包完整性之前，需要准备如下文件。

> ![icon note](public_sys-resources/icon-note.gif) **说明** 
> 以下软件包`BoostKit-boostcore-kdnn_xxx.zip`中占位符`xxx`需按实际版本替换。

- 软件包：BoostKit-boostcore-kdnn_xxx.zip。
- 校验文件：同名的.sha256文件。

文件完整性校验操作步骤如下：

1. 计算软件包的SHA256校验值。

   - Linux执行命令如下。

        ```bash
      sha256sum BoostKit-boostcore-kdnn_xxx.zip
      ```

   - Windows执行命令如下。

        ```bash
       certutil -hashfile BoostKit-boostcore-kdnn_xxx.zip SHA256
       ```

      命令执行完成后，输出校验值。

2. 对比步骤1计算的校验值与校验文件中的SHA256值是否一致。

   如果校验值一致说明文件完整，如果校验值不一致则可以确认文件完整性已被破坏，需要重新获取。

#### 安装KDNN软件包

- 将KDNN软件包解压缩，并使用rpm -ivh命令安装。

  ```bash
  unzip BoostKit-boostcore-kdnn_3.3.0.zip
  rpm -ivh boostcore-kdnn-3.3.0-1.aarch64.rpm
  ```

  安装后，头文件位于`/usr/local/kdnn/include`，静态库位于`/usr/local/kdnn/lib`。

## 编译安装Onnxruntime

1. 获取ONNX Runtime开源代码。假设安装路径为`/path/to`。

   ```bash
   cd /path/to
   git clone --branch v1.26.0 https://github.com/microsoft/onnxruntime.git
   ```

2. 下载并应用Patch。

   ```bash
   git clone --branch master https://gitcode.com/boostkit/onnxruntime.git kp_onnxruntime
   cd onnxruntime
   git apply ../kp_onnxruntime/patches/001-boostsra-onnxruntime-1.0.0-graph-optimization.patch
   git apply ../kp_onnxruntime/patches/002-boostsra-onnxruntime-1.0.0-parallel-executor.patch
   ```

3. 将KDNN头文件和静态库放入生成的Onnxruntime源码。

   ```bash
   export ORT_PATH=/path/to/onnxruntime
   mkdir -p $ORT_PATH/onnxruntime/core/kdnn/out
   cp -r /usr/local/kdnn/include $ORT_PATH/onnxruntime/core/kdnn/out
   cp -r /usr/local/kdnn/lib $ORT_PATH/onnxruntime/core/kdnn/out
   ```

4. 编译安装onnxruntime。

   ```bash
   # 编译onnxruntime
   ./build.sh --config Release --build_shared_lib --parallel \
     --compile_no_warning_as_error --skip_submodule_sync \
     --allow_running_as_root --skip_tests --build_wheel \
     --cmake_extra_defines onnxruntime_USE_KDNN=ON onnxruntime_USE_SVE=ON \
     --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_Protobuf=ON \
     --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
   # 安装onnxruntime，以下安装包以python3.13下为例
   pip install build/Linux/Release/dist/onnxruntime-1.26.0-cp313-cp313-linux_aarch64.whl
   ```

   其中：

   - `onnxruntime_USE_KDNN=ON`：编译并注册KDNN内部算子和MHA图融合。
   - `onnxruntime_USE_SVE=ON`：启用ORT/目标平台所需的SVE构建配置。
   - contrib ops必须保持启用；不能使用 `onnxruntime_DISABLE_CONTRIB_OPS=ON`。

## 构建安装ANNC

1. 安装ANNC前置依赖。

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

2. 获取ANNC源码。

   ```bash
   cd /path/to
   git clone --branch dev https://gitcode.com/openeuler/ANNC.git
   ```

3. 生成onnx protobuf。

   ```bash
   cd /path/to/ANNC
   export ANNC_PATH=/path/to/ANNC
   ./frontends/onnx/protos/onnx_protos_minimal.sh -d $ANNC_PATH
   ```

4. 配置并构建ORT加速栈。

   ```bash
   export ORT_PATH=/path/to/onnxruntime
   export ANNC_ONNXRUNTIME_INCLUDE_DIR=$ORT_PATH/include/onnxruntime/core/session
   export ANNC_EP_ORT_BUILD_TREE=$ORT_PATH/build/Release
   export ANNC_MLAS_INCLUDE_DIR=$ORT_PATH/onnxruntime/core/mlas/inc

   ./build.sh --no-install-deps \
      --build-type Release --install-prefix $ANNC_PATH/install --enable-ort
   ```

5. 编译并安装ANNC。

   ```bash
   export ANNC_PATH=/path/to/ANNC
   ninja -C $ANNC_PATH/build -j16
   ninja -C $ANNC_PATH/build install
   ```

   `j16`代表编译时的CPU核心数，用户实际编译时可以按照自己容器CPU配额取值。
   ANNC安装成功后，可以在`/path/to/ANNC/install`路径下看到如下关键产物。

   ```bash
   /bin/annc-onnx-pipeline
   /bin/annc-onnx2atir
   /bin/annc
   /bin/annc-opt
   /bin/annc-asm
   /lib64/libannc_ep.so
   /lib64/libANNC*.a
   ```

## 安装验证

安装完成后，运行以下python脚本，可以验证ONNX Runtime以及KDNN库、ANNC库是否安装成功。

```python
import onnxruntime as ort
import os
import ctypes
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

ctypes.CDLL(
    os.path.join(os.path.dirname(ort.__file__), "capi", "libonnxruntime.so.1.26.0"),
    mode=ctypes.RTLD_GLOBAL)
# 注意匹配如下的/path/to/ANNC/install/lib64/libannc_ep.so路径
rtpy.register_execution_provider_library(
    "AnncExecutionProvider", "/path/to/ANNC/install/lib64/libannc_ep.so")  

print(ort.__version__)    # 期望输出Onnxruntime版本1.26.0
print(ort.get_available_providers())   # 期望看到 ['KDNNExecutionProvider', 'CPUExecutionProvider', 'AnncExecutionProvider']
```

执行上述脚本后，如果输出如下内容，说明ONNX Runtime以及KDNN库、ANNC库安装成功。

```output
1.26.0
['KDNNExecutionProvider', 'CPUExecutionProvider', 'AnncExecutionProvider']
```

## 修订记录

| 文档版本 | 发布日期  | 修订记录  |
| ----- | ----- | ----- |
| 01 | 2026-12-30 | 第一次正式发布。 |
