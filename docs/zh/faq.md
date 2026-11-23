# 常见问题和故障案例

本文提供ONNX Runtime的常见问题和故障案例，涵盖安装ONNX Runtime等类别内容。

## 安装ONNX Runtime相关

### 编译ONNX Runtime时提示Target "onnxruntime_pybind11_state" not found

#### 问题现象描述

源码方式编译ONNX Runtime时，提示Target "onnxruntime_pybind11_state" not found，具体信息如下所示。

```bash
CMake Error at onnxruntime_python.cmake:214 (target_link_libraries):
  Target "onnxruntime_pybind11_state" links to:

    Python::NumPy

  but the target was not found.  Possible reasons include:

    * There is a typo in the target name.
    * A find_package call is missing for an IMPORTED target.
    * An ALIAS target is missing.
```

#### 关键过程、根本原因分析

此问题是由于系统中没有安装python3-devel导致build软连接所指向的依赖包不存在。

#### 结论、解决方案及效果

在系统中安装python3-devel。

```bash
# 确认没装python3-devel
rpm -q python3-devel

# 安装
yum install -y python3-devel
```

安装完python3-devel后，重新执行编译，预期不会再报错。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| ------ | ------ | ------ |
| 01 | 2026-12-30 | 第一次正式发布。|
