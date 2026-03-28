###### SetMaskCount

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

设置mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入计算数据量，实际迭代次数由Vector计算单元自动推断。

mask模式分为Counter模式和Normal模式，两种模式的概念和使用场景请参考12.5 如何使用掩码操作API。

## 函数原型

```cpp
__aicore__ inline void SetMaskCount()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

设置为Counter模式的场景需要在矢量计算使用完之后调用 `SetMaskNorm` 将mask模式恢复为Normal模式。

## 调用示例

请参考Counter模式调用示例。
