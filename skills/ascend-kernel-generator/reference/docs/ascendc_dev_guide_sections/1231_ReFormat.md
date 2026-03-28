###### ReFormat

## 支持的产品型号

- Atlas 训练系列产品
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品
- Atlas 推理系列产品

## 功能说明

该函数不改变 tensor 数据的值，在指定 format 和输入 x 的维度相同时，将输入数据格式设置为目标 format。

具体功能是将用户输入 tensor 的 viewFormat、originalFormat、storageFormat 统一为指定的 format。

## 函数原型

```cpp
const aclTensor *ReFormat(const aclTensor *x, const op::Format &format,
                          aclOpExecutor *executor=nullptr)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| x | 输入 | 需要被转换的 tensor。数据类型支持 FLOAT16、FLOAT32、INT32、UINT32、INT8、UINT8。 |
| format | 输入 | 输入 tensor 要转换的目标 format。 |
| executor | 输入 | op 执行器，包含了算子计算流程。 |

## 返回值说明

返回数据格式为目标 format 后的 tensor。

## 约束说明

输入 tensor 的数据的维度需要与指定 format 的维度相同。

## 调用示例

```cpp
// 将输入 reformat 成 NCHW 格式
auto reformatInput = l0op::ReFormat(unsqueezedInput, Format::FORMAT_NCHW);
CHECK_RET(reformatInput != nullptr, nullptr);
```
