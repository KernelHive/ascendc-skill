##### GetInputConstData

## 函数功能

如果指定算子 Input 对应的节点为 Const 节点，可调用该接口获取 Const 节点的数据。

## 函数原型

```cpp
graphStatus GetInputConstData(const std::string &dst_name, Tensor &data) const
graphStatus GetInputConstData(const char_t *dst_name, Tensor &data) const
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名   | 输入/输出 | 描述                     |
|----------|-----------|--------------------------|
| dst_name | 输入      | 输入名称                 |
| data     | 输出      | 返回 Const 节点的数据 Tensor |

## 返回值

graphStatus 类型：

- 如果指定算子 Input 对应的节点为 Const 节点且获取数据成功，返回 `GRAPH_SUCCESS`
- 否则，返回 `GRAPH_FAILED`

## 异常处理

无。

## 约束说明

无。
