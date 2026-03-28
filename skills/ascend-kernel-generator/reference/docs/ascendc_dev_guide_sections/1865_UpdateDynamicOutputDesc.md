##### UpdateDynamicOutputDesc

## 函数功能

根据 name 和 index 的组合更新算子动态 Output 的 TensorDesc。

## 函数原型

```cpp
graphStatus UpdateDynamicOutputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc)
graphStatus UpdateDynamicOutputDesc(const char_t *name, uint32_t index, const TensorDesc &tensor_desc)
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名      | 输入/输出 | 描述                     |
|-------------|-----------|--------------------------|
| name        | 输入      | 算子动态 Output 的名称。 |
| index       | 输入      | 算子动态 Output 编号。   |
| tensor_desc | 输入      | TensorDesc 对象。        |

## 返回值

graphStatus 类型：更新动态 Output 成功，返回 GRAPH_SUCCESS，否则，返回 GRAPH_FAILED。

## 异常处理

无。

## 约束说明

无。
