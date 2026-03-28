##### FindOpByType

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

基于算子类型，获取缓存在 Graph 中的所有指定类型的 op 对象。

## 函数原型

> **须知**
>
> 数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

```cpp
graphStatus FindOpByType(const std::string &type, std::vector<ge::Operator> &ops) const
graphStatus FindOpByType(const char_t *type, std::vector<ge::Operator> &ops) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| type | 输入 | 需要获取的算子类型。 |
| ops | 输出 | 返回用户所需要的 op 对象。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS(0)：成功。<br>其他值：失败。 |

## 约束说明

无
