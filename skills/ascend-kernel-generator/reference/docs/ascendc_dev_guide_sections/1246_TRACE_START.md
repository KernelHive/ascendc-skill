#### TRACE_START

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | √ |

## 功能说明

通过CAModel进行算子性能仿真时，可对算子任意运行阶段打点，从而分析不同指令的流水图，以便进一步性能调优。

用于表示起始位置打点，一般与 `TRACE_STOP` 配套使用。

## 函数原型

```c
#define TRACE_START(apid)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| apid | 输入 | 当前预留了十个用户自定义的类型：<br>● 0x0：USER_DEFINE_0<br>● 0x1：USER_DEFINE_1<br>● 0x2：USER_DEFINE_2<br>● 0x3：USER_DEFINE_3<br>● 0x4：USER_DEFINE_4<br>● 0x5：USER_DEFINE_5<br>● 0x6：USER_DEFINE_6<br>● 0x7：USER_DEFINE_7<br>● 0x8：USER_DEFINE_8<br>● 0x9：USER_DEFINE_9 |

## 返回值说明

无

## 约束说明

- `TRACE_START`/`TRACE_STOP` 需配套使用，若Trace图上未显示打点，则说明两者没有配对。
- 不支持跨核使用，例如 `TRACE_START` 在AI Cube打点，则 `TRACE_STOP` 打点也需要在AI Cube上，不能在AI Vector上。
- 宏支持所有的产品型号，但实际调用时需与调测工具支持的型号保持一致。

## 调用示例

在Kernel代码中特定指令位置打上 `TRACE_START`/`TRACE_STOP`：

```c
TRACE_START(0x2);
Add(zLocal, xLocal, yLocal, dataSize);
TRACE_STOP(0x2);
```
