#### ELMTWISE_INFER_SHAPEANDTYPE

## 函数功能

提供公共函数宏封装，供算子开发者开发 InferShape 函数。该函数基于输入的 shape 和 dtype，设置输出的 shape 和 dtype。

例如，输入 shape 为 (1,2,3,4)，dtype 为 float，则该宏会设置算子的输出 shape 为 (1,2,3,4)，输出 dtype 为 float。

## 函数原型

```c
ELMTWISE_INFER_SHAPEANDTYPE(in_name, out_name)
```

## 约束说明

无。

## 参数说明

| 参数名   | 输入/输出 | 描述       |
|----------|-----------|------------|
| in_name  | 输入      | 算子输入。 |
| out_name | 输入      | 算子输出。 |

## 返回值

执行成功或失败。

## 调用示例

```c
COMMON_INFER_FUNC_REG(DiagD, ELMTWISE_INFER_SHAPEANDTYPE("assist", "y"));
```
