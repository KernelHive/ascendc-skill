##### GetDynamicSubgraphBuilder

## 函数功能

根据子图名称和子图索引获取算子对应的动态输入子图的构建函数对象。

## 函数原型

```cpp
SubgraphBuilder GetDynamicSubgraphBuilder(const std::string &name, uint32_t index) const
SubgraphBuilder GetDynamicSubgraphBuilder(const char_t *name, uint32_t index) const
```

## 须知

数据类型为 `string` 的接口后续版本会废弃，建议使用数据类型为非 `string` 的接口。

## 参数说明

| 参数  | 输入/输出 | 描述           |
|-------|-----------|----------------|
| name  | 输入      | 子图名         |
| index | 输入      | 同名子图的索引 |

## 返回值

`SubgraphBuilder` 对象。

## 异常处理

无。

## 约束说明

无。
