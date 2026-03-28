##### GetOpType

## 函数功能

获取算子类型。

## 函数原型

```cpp
std::string GetOpType() const
graphStatus GetOpType(AscendString &type) const
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述     |
|--------|-----------|----------|
| type   | 输出      | 算子类型 |

## 返回值

graphStatus 类型：

- GRAPH_SUCCESS：代表成功
- GRAPH_FAILED：代表失败

## 异常处理

无。

## 约束说明

无。
