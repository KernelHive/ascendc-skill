##### operator()

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

获取本 LocalTensor 的第 offset 个变量的引用。用于左值，相当于 SetValue 接口；用于右值，相当于 GetValue 接口。

## 函数原型

```cpp
__aicore__ inline __inout_pipe__(S) __ubuf__ PrimType& operator()(const uint32_t offset) const
```

## 参数说明

**表 15-36 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| offset | 输入 | LocalTensor 下标索引。 |

## 返回值说明

返回指定索引位置的元素的 PrimType 类型引用。

PrimType 定义如下：

```cpp
// PrimT 用于从 T 中提取基础数据类型：T 传入基础数据类型，直接返回数据类型；T 传入为 TensorTrait 类型时萃取 TensorTrait 中的 LiteType 基础数据类型
using PrimType = PrimT<T>;
```

## 约束说明

无

## 调用示例

参考调用示例。
