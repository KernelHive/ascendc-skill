##### SetAddrWithOffset

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

设置带有偏移的 Tensor 地址。用于快速获取定义一个 Tensor，同时指定新 Tensor 相对于旧 Tensor 首地址的偏移。偏移的长度为旧 Tensor 的元素个数。

## 函数原型

```cpp
template <typename T1>
__aicore__ inline void SetAddrWithOffset(LocalTensor<T1> &src, uint32_t offset)
```

## 参数说明

**表 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| src | 输入 | 基础地址的 Tensor，将该 Tensor 的地址作为基础地址，设置偏移后的 Tensor 地址。 |
| offset | 输入 | 偏移的长度，单位为元素。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

参考调用示例。
