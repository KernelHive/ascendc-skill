###### GetSocVersion

## 功能说明
获取当前硬件平台版本型号。

## 函数原型
```cpp
SocVersion GetSocVersion(void) const
```

## 参数说明
无

## 返回值说明
当前硬件平台版本型号的枚举类。该枚举类和AI处理器型号的对应关系请通过CANN软件安装后文件存储路径下`include/tiling/platform/platform_ascendc.h`头文件获取。

AI处理器的型号请通过如下方式获取：

- **针对如下产品型号**：在安装昇腾AI处理器的服务器执行`npu-smi info`命令进行查询，获取Name信息。实际配置值为`AscendName`，例如Name取值为`xxxyy`，实际配置值为`Ascendxxxyy`。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品
  - Atlas 200I/500 A2 推理产品
  - Atlas 推理系列产品
  - Atlas 训练系列产品

- **针对如下产品型号**：在安装昇腾AI处理器的服务器执行`npu-smi info -t board -i id -c chip_id`命令进行查询，获取Chip Name和NPU Name信息，实际配置值为`Chip Name_NPU Name`。例如Chip Name取值为`Ascendxxx`，NPU Name取值为`1234`，实际配置值为`Ascendxxx_1234`。其中：
  - `id`：设备id，通过`npu-smi info -l`命令查出的NPU ID即为设备id。
  - `chip_id`：芯片id，通过`npu-smi info -m`命令查出的Chip ID即为芯片id。
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 约束说明
无

## 调用示例
```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context) {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto socVersion = ascendcPlatform.GetSocVersion();
  // 根据所获得的版本型号自行设计Tiling策略
  // ASCENDXXX请替换为实际的版本型号
  if (socVersion == platform_ascendc::SocVersion::ASCENDXXX) {
    // ...
  }
  return ret;
}
```
