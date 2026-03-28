###### TransData

## 支持的产品型号

- Atlas 训练系列产品
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品
- Atlas 推理系列产品

## 功能说明

该函数不改变 tensor 数据的值，实现对 tensor 数据格式的转换。

具体功能是将用户输入 tensor 的 format 转换为指定的 dstPrimaryFormat。

## 函数原型

```cpp
const aclTensor *TransData(const aclTensor *x, 
                           op::Format dstPrimaryFormat,
                           int64_t groups, 
                           aclOpExecutor *executor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| x | 输入 | 待转换的 tensor。数据类型支持 FLOAT16、FLOAT32、INT32、UINT32、INT8、UINT8。 |
| dstPrimaryFormat | 输入 | 输入 tensor 要转换的目标 format。 |
| groups | 输入 | 分组参数，用于分组转换时传入。数据类型支持 INT64。 |
| executor | 输入 | op 执行器，包含了算子计算流程。 |

## 返回值说明

返回数据格式为 dstPrimaryFormat 的 tensor。

## 约束说明

当输入 tensor 数据类型为 FLOAT32、INT32、UINT32 时，C0 只能按照 8 处理。

## 调用示例

```cpp
// 将输出的格式从 NC1HWC0 转换成 NCHW
auto transGradInput = l0op::TransData(gradInputNC1HWC0, 
                                      Format::FORMAT_NCHW, 
                                      params.groups,
                                      executor);
CHECK_RET(transGradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
```
