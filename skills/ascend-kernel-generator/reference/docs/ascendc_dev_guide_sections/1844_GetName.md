##### GetName

## 函数功能
获取算子名称。

## 函数原型
```cpp
std::string GetName() const
graphStatus GetName(AscendString &name) const
```

> **须知**
> 数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述       |
|--------|-----------|------------|
| name   | 输出      | 算子名称。 |

## 返回值
graphStatus 类型：
- GRAPH_SUCCESS：代表成功
- GRAPH_FAILED：代表失败

## 异常处理
无。

## 约束说明
无。
