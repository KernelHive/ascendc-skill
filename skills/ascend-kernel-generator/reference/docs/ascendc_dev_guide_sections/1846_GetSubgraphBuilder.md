##### GetSubgraphBuilder

## 函数功能

根据子图名称获取算子对应的子图构建的函数对象。

## 函数原型

```cpp
SubgraphBuilder GetSubgraphBuilder(const std::string &name) const
SubgraphBuilder GetSubgraphBuilder(const char_t *name) const
```

## 须知

数据类型为 `string` 的接口后续版本会废弃，建议使用数据类型为非 `string` 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述     |
|--------|-----------|----------|
| name   | 输入      | 子图名称 |

## 返回值

`SubgraphBuilder` 对象。

## 异常处理

无。

## 约束说明

无。
