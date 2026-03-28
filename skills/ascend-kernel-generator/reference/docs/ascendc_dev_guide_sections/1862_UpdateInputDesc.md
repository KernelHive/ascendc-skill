##### UpdateInputDesc

## 函数功能

根据算子 Input 名称更新 Input 的 TensorDesc。

## 函数原型

```cpp
graphStatus UpdateInputDesc(const std::string &name, const TensorDesc &tensor_desc)
graphStatus UpdateInputDesc(const char_t *name, const TensorDesc &tensor_desc)
```

## 须知

数据类型为 `string` 的接口后续版本会废弃，建议使用数据类型为非 `string` 的接口。

## 参数说明

| 参数名      | 输入/输出 | 描述                 |
|-------------|-----------|----------------------|
| name        | 输入      | 算子 Input 名称。    |
| tensor_desc | 输入      | TensorDesc 对象。    |

## 返回值

`graphStatus` 类型：更新 TensorDesc 成功，返回 `GRAPH_SUCCESS`，否则，返回 `GRAPH_FAILED`。

## 异常处理

| 异常场景       | 说明                               |
|----------------|------------------------------------|
| 无对应 name 输入 | 函数提前结束，返回 `GRAPH_FAILED`。 |

## 约束说明

无。
