###### Slice

## 支持的产品型号

- Atlas 训练系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas 推理系列产品

## 功能说明

从输入 tensor 中提取想要的切片。

## 函数原型

```cpp
const aclTensor *Slice(const aclTensor *x, const aclIntArray *offsets, const aclIntArray *size, aclOpExecutor *executor)
```

```cpp
const aclTensor *Slice(const aclTensor *x, const aclTensor *y, const aclTensor *offset, const aclTensor *size, aclOpExecutor *executor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| x | 输入 | 输入 tensor，数据类型支持 FLOAT16、FLOAT、BOOL、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、BFLOAT16、UINT64。数据格式支持 ND。<br>**说明**：BFLOAT16 适用于如下产品型号：<br>- Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>- Atlas A3 训练系列产品/Atlas A3 推理系列产品 |
| y | 输出 | 切片后的输出 tensor，数据类型支持 FLOAT16、FLOAT、BOOL、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、BFLOAT16、UINT64。数据格式支持 ND。<br>**说明**：BFLOAT16 适用于如下产品型号：<br>- Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>- Atlas A3 训练系列产品/Atlas A3 推理系列产品 |
| offsets | 输入 | const aclIntArray* 类型，表示输入 x 在各个维度切片的起始位置，其形状为 x 的维度。数据类型支持 INT32、INT64。数据格式支持 ND。 |
| offset | 输入 | const aclTensor* 类型，表示输入 x 在各个维度切片的起始位置，其形状为 x 的维度。数据类型支持 INT32、INT64。数据格式支持 ND。 |
| size | 输入 | 输入 x 的各个维度切片的大小，其形状为 x 的维度。支持 aclIntArray*、aclTensor* 类型。数据类型支持 INT32、INT64。数据格式支持 ND。 |
| executor | 输入 | op 执行器，包含了算子计算流程。 |

## 返回值说明

返回类型和输入 tensor 一样、shape 为 size 的 tensor。

## 约束说明

无

## 调用示例

```cpp
// 调用 l0op::Slice 对每一块进行处理
auto sliceRes = l0op::Slice(self, offsetArray, sizeArray, executor);
```

```cpp
// 调用 l0op::Slice 对每一块进行处理
auto sliceRes = l0op::Slice(xTensor, yTensor, offsetTensor, sizeTensor, executor);
```
