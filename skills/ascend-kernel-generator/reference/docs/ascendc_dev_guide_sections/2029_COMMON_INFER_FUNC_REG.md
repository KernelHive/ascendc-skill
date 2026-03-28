#### COMMON_INFER_FUNC_REG

## 函数功能
注册算子的 InferShape 函数。

与 `15.2.3.69 INFER_FUNC_REG` 的区别是，此函数注册的 InferShape 函数入参为 Operator 基类而非子类，此接口支持多算子共用同一个 InferShape 函数。

## 函数原型
```cpp
COMMON_INFER_FUNC_REG(op_name, x)
```

该函数内部会自动调用 `COMMON_INFER_VERIFY_FUNC(x)`，`COMMON_INFER_VERIFY_FUNC(x)` 函数中的 `x` 为指向 `COMMON_INFER_FUNC_REG(op_name, x)` 中 “x” 的指针。

## 约束说明
无。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| op_name | 输入 | 算子类型。 |
| x | 输入 | InferShape 函数名，和 `15.2.3.64 IMPLEMT_COMMON_INFERFUNC` 的 InferShape 函数名保持一致。 |

## 返回值
无。
