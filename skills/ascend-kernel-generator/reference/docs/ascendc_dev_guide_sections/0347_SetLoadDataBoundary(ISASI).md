##### SetLoadDataBoundary(ISASI)

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 AI Core | √ |
| Atlas 推理系列产品 Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

设置 Load3D 时 A1/B1 边界值。

如果 Load3D 指令在处理源操作数时，源操作数在 A1/B1 上的地址超出设置的边界，则会从 A1/B1 起始地址开始读取数据。

## 函数原型

```cpp
__aicore__ inline void SetLoadDataBoundary(uint32_t boundaryValue)
```

## 参数说明

**表 参数说明**

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| boundaryValue | 输入 | 边界值。<br>Load3Dv1 指令：单位是 32 字节。<br>Load3Dv2 指令：单位是字节。 |

## 约束说明

- 用于 Load3Dv1 时，boundaryValue 的最小值是 16（单位：32 字节）；用于 Load3Dv2 时，boundaryValue 的最小值是 1024（单位：字节）。
- 如果使用 SetLoadDataBoundary 接口设置了边界值，配合 Load3D 指令使用时，Load3D 指令的 A1/B1 初始地址要在设置的边界内。
- 如果 boundaryValue 设置为 0，则表示无边界，可使用整个 A1/B1。
- 操作数地址对齐要求请参见通用地址对齐约束。

## 调用示例

参考调用示例。
