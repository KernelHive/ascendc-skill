##### SetValue

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

设置 LocalTensor 中的某个值。

该接口仅在 LocalTensor 的 TPosition 为 VECIN/VECCALC/VECOUT 时支持。

## 函数原型

```cpp
template <typename T1> __aicore__ inline __inout_pipe__(S)
void SetValue(const uint32_t index, const T1 value) const
```

## 参数说明

**表 15-33 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| index | 输入 | LocalTensor 索引，单位为元素。 |
| value | 输入 | 待设置的数值。 |

## 返回值说明

无

## 约束说明

不要大量使用 SetValue 对 LocalTensor 进行赋值，会使性能下降。若需要大批量赋值，请根据实际场景选择数据填充基础 API 接口或数据填充高阶 API 接口，以及在需要生成递增数列的场景，选择 ArithProgression。

## 调用示例

参考调用示例。
