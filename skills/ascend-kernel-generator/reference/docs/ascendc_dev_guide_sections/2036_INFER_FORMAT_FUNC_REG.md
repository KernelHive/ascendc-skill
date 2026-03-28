#### INFER_FORMAT_FUNC_REG

## 函数功能

注册算子的 InferFormat 实现。

GE 会在整图的 Shape 与 Dtype 推导前后分别调用一次整图的 InferFormat，过程中会分别调用各个算子的 InferFormat 函数。如果算子没有注册 InferFormat 函数，GE 将使用默认的推导函数，即输出的 Format 等于输入的 Format。

## 函数原型

```c
#define INFER_FORMAT_FUNC_REG(op_name, x) \
__INFER_FORMAT_FUNC_REG_IMPL__(op_name, INFER_FORMAT_FUNC(op_name, x), __COUNTER__)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| op_name | 输入 | 算子类型。 |
| x | 输入 | InferFormat 函数名，使用 `IMPLEMT_INFERFORMAT_FUNC` 中的 `func_name`。 |

## 返回值

无。

## 约束和限制说明

无。

## 调用示例和相关 API

```c
INFER_FORMAT_FUNC_REG(Transpose, TransposeInferFormat);
```
