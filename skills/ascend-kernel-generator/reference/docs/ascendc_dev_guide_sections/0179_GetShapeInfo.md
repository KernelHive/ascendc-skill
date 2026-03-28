##### GetShapeInfo

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 AI Core | √ |
| Atlas 推理系列产品 Vector Core | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

获取 GlobalTensor 的 shape 信息。

> **注意**：Shape 信息没有默认值，只有通过 `SetShapeInfo` 设置过 Shape 信息后，才可以调用该接口获取正确的 ShapeInfo。

## 函数原型

```cpp
__aicore__ inline ShapeInfo GetShapeInfo() const
```

## 参数说明

无。

## 返回值说明

GlobalTensor 的 shape 信息，ShapeInfo 类型。

## 约束说明

无。
