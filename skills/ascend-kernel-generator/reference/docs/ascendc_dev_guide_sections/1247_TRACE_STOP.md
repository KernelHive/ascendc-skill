#### TRACE_STOP

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

## 功能说明

通过CAModel进行算子性能仿真时，可对算子任意运行阶段打点，从而分析不同指令的流水图，以便进一步性能调优。

用于表示终止位置打点，一般与`TRACE_START`配套使用。

## 函数原型

```c
#define TRACE_STOP(apid)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| apid | 输入 | 取值需与`TRACE_START`参数取值保持一致，否则影响打点结果。 |

## 返回值说明

无

## 约束说明

- `TRACE_START`/`TRACE_STOP`需配套使用，若Trace图上未显示打点，则说明两者没有配对。
- 不支持跨核使用，例如`TRACE_START`在AI Cube打点，则`TRACE_STOP`打点也需要在AI Cube上，不能在AI Vector上。
- 宏支持所有的产品型号，但实际调用时需与调测工具支持的型号保持一致。

## 调用示例

在Kernel代码中特定指令位置打上`TRACE_START`/`TRACE_STOP`：

```c
TRACE_START(0x1);
DataCopy(zGm, zLocal, this->totalLength);
TRACE_STOP(0x1);
```
