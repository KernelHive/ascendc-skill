#### PrintTimeStamp

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

提供时间戳打点功能，用于在算子Kernel代码中标记关键执行点，调用后会打印如下信息：

- **descId**：用户自定义标识符，用于区分不同打点位置
- **rsv**：保留值，默认为0，无需关注
- **timeStamp**：当前系统cycle数，用于计算时间差，时间换算规则可参考15.1.4.6.10 GetSystemCycle(ISASI)
- **pcPtr**：pc指针数值，若无特殊需求，用户无需关注

打印示例如下：
```
descId is 65577, rsv is 0, timeStamp is 13806084506158, pcPtr is 20619064414544.
```

> **注意**
> 
> 该功能主要用于调试和性能分析，开启后会对算子性能产生一定影响，生产环境建议关闭。
> 
> 默认情况下，该功能关闭，开发者可以按需通过如下方式开启打点功能。

### Kernel直调工程

修改cmake目录下的npu_lib.cmake文件，在ascendc_compile_definitions命令中增加`-DASCENDC_TIME_STAMP_ON`，打开时间戳打点功能，示例如下：

```cmake
ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
-DASCENDC_TIME_STAMP_ON
)
```

> CPU域调试时，不支持该功能。

### 自定义算子工程

修改算子工程op_kernel目录下的CMakeLists.txt文件，首行增加编译选项`-DASCENDC_TIME_STAMP_ON`，打开时间戳打点功能，示例如下：

```cmake
add_ops_compile_options(ALL OPTIONS -DASCENDC_TIME_STAMP_ON)
```

## 函数原型

```cpp
void PrintTimeStamp(uint32_t descId)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| descId | 输入 | 用户自定义标识符（自定义数字），用于区分不同打点位置 |

> **注意**
> 
> [0, 0xffff]是预留给Ascend C内部各个模块使用的id值，用户自定义的descId建议使用大于0xffff的数值。

## 返回值说明

无

## 约束说明

- 该功能仅用于NPU上板调试，且仅在如下场景支持：
  - 通过Kernel直调方式调用算子
  - 通过单算子API调用方式调用算子
  - 间接调用单算子API(aclnnxxx)接口：PyTorch框架单算子直调的场景
- 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印

## 调用示例

```cpp
AscendC::PrintTimeStamp(65577);
```

打印结果如下：
```
opType=AddCustom, DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, block_remain_len=1047136, block_initial_space=1048576, rsv=0, magic=5aa5bccd
...// 一些框架内部的打点信息
descId is 65577, rsv is 0, timeStamp is 13806084506158, pcPtr is 20619064414544.
```
