###### ViewCopy

## 支持的产品型号

- Atlas 训练系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas 200I/500 A2 推理产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas 推理系列产品

## 功能说明

该函数为公共的L0接口，作用是将连续tensor搬运到连续或非连续tensor上。

由于L2级API的输出tensor可能是非连续的，需要通过该API将连续的tensor搬运到非连续的tensor上。

输出tensor可以是连续或非连续的，接口内部会兼容处理，但输入要求是连续的。

## 函数原型

```cpp
const aclTensor *ViewCopy(const aclTensor *x, const aclTensor *y, aclOpExecutor *executor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| x | 输入 | 输入tensor。数据类型和数据格式不限制。输入必须保证是连续内存数据。 |
| y | 输出 | 输出tensor。数据类型和数据格式不限制，但数据类型、ViewShape和数据格式要求与x一致。 |
| executor | 输入 | op执行器，包含了算子计算流程。 |

## 返回值说明

若转换成功，返回为输出Tensor，若转换失败，则返回nullptr。

## 约束说明

要求输入tensor是连续的tensor。

## 调用示例

```cpp
// 固定写法，创建OpExecutor
auto uniqueExecutor = CREATE_EXECUTOR();
// 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
auto viewCopyResult = l0op::ViewCopy(absResult, out, executor);
```
