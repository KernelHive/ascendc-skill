##### UpdateOutputDesc

## 函数功能

根据算子 Output 名称更新 Output 的 TensorDesc。

## 函数原型

```cpp
graphStatus UpdateOutputDesc(const std::string &name, const TensorDesc &tensor_desc)
graphStatus UpdateOutputDesc(const char_t *name, const TensorDesc &tensor_desc)
graphStatus UpdateOutputDesc(const uint32_t index, const TensorDesc &tensor_desc)
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名      | 输入/输出 | 描述                 |
|-------------|-----------|----------------------|
| name        | 输入      | 算子 Output 名称     |
| tensor_desc | 输入      | TensorDesc 对象      |
| index       | 输入      | 算子 Output 的序号   |

## 返回值

graphStatus 类型：更新 TensorDesc 成功，返回 `GRAPH_SUCCESS`，否则，返回 `GRAPH_FAILED`。

## 异常处理

无。

## 约束说明

无。
