##### operator()

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

获取某个索引位置的元素的引用。

与15.1.3.1 LocalTensor的operator()类似。

## 函数原型

```cpp
__aicore__ inline __inout_pipe__(S) __gm__ PrimType& operator()(const uint64_t offset) const
```

## 参数说明

**表 15-50 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| offset | 输入 | GlobalTensor的下标索引。 |

## 返回值说明

返回指定索引位置的元素的PrimType类型引用。

PrimType定义如下：

```cpp
// PrimT用于从T中提取基础数据类型：T传入基础数据类型，直接返回数据类型；T传入为TensorTrait类型时萃取TensorTrait中的LiteType基础数据类型
using PrimType = PrimT<T>;
```

## 约束说明

无。
