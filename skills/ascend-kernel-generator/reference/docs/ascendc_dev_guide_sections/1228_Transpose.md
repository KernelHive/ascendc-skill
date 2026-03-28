###### Transpose

## 支持的产品型号

- Atlas 训练系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas 推理系列产品

## 功能说明

该函数不改变 tensor 数据的值，而是对 tensor 进行转置操作。

通过 transpose 算子，可以改变 tensor 在不同维度上的排列顺序，实现对 tensor 的维度重排。

具体的是将用户传入的输入 tensor x 的 shape 按指定维度的排列顺序 perm 进行转置并输出。

## 函数原型

- **输入和输出为不同地址**
  ```cpp
  const aclTensor *Transpose(const aclTensor *x, const aclTensor *y, const aclTensor *perm, aclOpExecutor *executor)
  ```

- **输入和输出同一地址**
  ```cpp
  const aclTensor *Transpose(const aclTensor *x, const aclIntArray *perm, aclOpExecutor *executor)
  ```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| x | 输入/输出 | 原始输入 tensor。输入 tensor x 需是连续内存数据。<br>数据类型支持：FLOAT16、FLOAT、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL、BFLOAT16。<br>数据格式支持：ND。<br><br>**说明**<br>BFLOAT16 适用于如下产品型号：<br>- Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>- Atlas A3 训练系列产品/Atlas A3 推理系列产品 |
| y | 输出 | 转置后输出 tensor。数据类型和数据格式同 x。 |
| perm | 输入 | 整型数组，代表输入 tensor x 的维度，支持 aclIntArray*、aclTensor* 类型。<br>最多支持 8 维转置，取值需在 [0，x 的维度数量-1] 范围内。<br>数据类型支持：INT32、INT64。<br>数据格式支持：ND。 |
| executor | 输入 | op 执行器，包含了算子计算流程。 |

## 返回值说明

转置成功，返回转置后的 tensor；转置失败，返回 nullptr。

## 约束说明

- 最多支持 8 维转置，即输入 x 和 perm 的 dim 至多为 8。
- 输入 x 和 perm 的 dim 维度需要一致。

## 调用示例

```cpp
// 固定写法，创建 OpExecutor，参数检查
auto uniqueExecutor = CREATE_EXECUTOR();
CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

// 固定写法，将输入 self 转换成连续的 tensor
auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

int64_t dims = selfContiguous->GetViewShape().GetDimNum();
int64_t valuePerm[dims] = {0, 2, 1, 3}; // 表示对原始 4 维的中间 2 维做转置，即交换 1 轴和 2 轴

auto perm = executor->AllocIntArray(valuePerm, dims);
selfContiguous = l0op::Transpose(selfContiguous, perm, uniqueExecutor.get());
```
