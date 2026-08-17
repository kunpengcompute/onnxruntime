# 安装指南

## 已验证环境

为保证您可以顺利安全地使用鲲鹏ONNX Runtime特性，请确保所使用的环境信息在已验证环境范围内。

**硬件要求**

已验证的硬件环境如[**表 1** 硬件要求](#硬件要求)所示。

**表 1** 硬件要求<a id="硬件要求"></a>

| 项目  | 说明   |
| --- | -------------- |
| CPU | 鲲鹏950 7592C处理器 |

**操作系统要求**

已验证的操作系统如[**表 2** 操作系统要求](#操作系统要求)所示。

**表 2** 操作系统<a id="操作系统要求"></a>

| 项目  | 版本    | 说明  | 下载地址  |
| --- | ----------- | ------------- | -------------- |
| OS  | openEuler 24.03 LTS SP3 | 如果是全新安装操作系统，可选择“Minimal Install”安装方式并勾选Development Tools套件，否则很多软件包需要手动安装。 | [获取链接](https://repo.openeuler.org/openEuler-24.03-LTS-SP3/ISO/aarch64/) |

**软件要求**

已验证的软件依赖环境如[**表 3** 软件要求](#软件要求)所示。

**表 3** 软件要求<a id="软件要求"></a>

| 项目      | 版本      | 说明     | 下载地址    |
| ------- | ------- | --- | ----------- |
| Python | 3.11.15 | Python是ONNX Runtime的构建过程中的辅助工具，起到自动下载依赖、配置环境等作用。 | 通过Yum源方式安装。 |
| CMake | 3.30.3 | CMake是ONNX Runtime的构建工具，要求CMake版本为3.28.3及以上。 | 通过Yum源方式安装。 |
| GCC/G++ | 12.3.1  | GCC（GNU Compiler Collection）是一种编程语言编译器。 | 通过Yum源方式安装。 |

## 编译安装

1. 获取ONNX Runtime开源代码。假设安装路径为“/path/to”。

   ```bash
   cd /path/to
   git clone --branch v1.26.0 https://github.com/microsoft/onnxruntime.git
   ```

2. 下载并应用Patch。

   ```bash
   git clone https://gitcode.com/boostkit/onnxruntime.git kp_onnxrumtime
   cd onnxruntime
   git apply ../kp_onnxrumtime/001-boostsra-onnxruntime-1.0.0-graph-optimization.patch
   git apply ../kp_onnxrumtime/002-boostsra-onnxruntime-1.0.0-parallel-executor.patch
   ```

3. 编译安装

   ```bash
   sh ./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 --allow_running_as_root --skip-keras-test --skip_tests --build_wheel
   ```

## 修订记录

| 发布日期 | 修订记录 |
| ---- | ---- |
| 2026-09-30 | 第一次正式发布。 |
