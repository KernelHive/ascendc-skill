###### Contiguous

## 支持的产品型号

- Atlas 训练系列产品
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas 200I / 500 A2 推理产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品
- Atlas 推理系列产品

## 功能说明

该函数为公共的 L0 接口，作用是将非连续 tensor 转换为连续 tensor。

由于 L2 级 API 的输入 tensor 可能是非连续的，而一般 L0 级算子只支持连续 tensor 作为输入，因此需要将 tensor 转为连续后再作为其他 L0 算子的输入。

输入 tensor 可以是连续的，接口内部会兼容处理。

## 函数原型

```cpp
const aclTensor *Contiguous(const aclTensor *x, aclOpExecutor *executor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| x | 输入 | 待转换的输入 tensor。数据类型和数据格式不限制。输入不要求是连续内存，但要求所表达的数据在 Storage 范围内。 |
| executor | 输入 | op 执行器，包含了算子计算流程。 |

## 返回值说明

若转换成功，则返回一个连续的 aclTensor；若失败，则返回 nullptr。

## 约束说明

要求输入 tensor 是合法的 tensor，Shape 和 Stride 所表示的数据在 Storage 大小范围内。

例如：shape=(2, 3), stride=(10, 30), storageSize=8，数据实际空间超过了 Storage 大小 8，该 Tensor 为非法，Contiguous 接口返回 nullptr。

## 调用示例

```cpp
// 固定写法，创建 OpExecutor
auto uniqueExecutor = CREATE_EXECUTOR();
// self 如果非连续，需要转换
auto selfContiguous = l0op::Contiguous(self, executor);
```
