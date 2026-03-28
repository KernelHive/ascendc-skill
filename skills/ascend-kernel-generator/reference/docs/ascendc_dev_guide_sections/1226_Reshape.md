###### Reshape

## 支持的产品型号

- Atlas 训练系列产品
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas 200I / 500 A2 推理产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品
- Atlas 推理系列产品

## 功能说明

该函数不改变算子 tensor 数据，只是将用户传入的输入 tensor `x` 的 shape 转换成该函数的第二个参数 `shape`。

## 函数原型

```cpp
const aclTensor *Reshape(const aclTensor *x, const op::Shape &shape,
                         aclOpExecutor *executor)

const aclTensor *Reshape(const aclTensor *x, const aclIntArray *shape,
                         aclOpExecutor *executor)
```

## 参数说明

| 参数      | 输入/输出 | 说明                                                                 |
|-----------|-----------|----------------------------------------------------------------------|
| x         | 输入      | 待转换的输入 tensor。数据类型和数据格式不限制。输入必须保证是连续内存数据。 |
| shape     | 输入      | 转换后的目标 shape，支持 `aclIntArray*`、`op::Shape`（即 `gert::Shape`）类型。数据类型和数据格式不限制。 |
| executor  | 输入      | op 执行器，包含了算子计算流程。                                      |

## 返回值说明

若 reshape 转换成功，则返回带有目标 shape 信息的 `aclTensor` 给调用者；若失败，则返回 `nullptr`。

## 约束说明

- reshape 转换成功的前提是 `x` 的 ShapeSize 需要和第二个参数 `shape` 的 ShapeSize 相等，所谓的 ShapeSize 举例如下：A 的 shape = (1, 3, 256, 256)，则 A 的 ShapeSize = 1 × 3 × 256 × 256。
- 当前不支持转换成空 tensor，所谓的空 tensor 即 shape 中包含 0。

## 调用示例

```cpp
void Func(const aclTensor *x, const op::Shape &shape, aclOpExecutor *executor) {
    auto ret = l0op::Reshape(x, shape, executor);
    return;
}
```
