##### GetAttrValue

## 函数功能

获取属性的取值。泛型类，承载具体的属性类型。

## 函数原型

```cpp
graphStatus GetAttrValue(int64_t &attr_value) const
graphStatus GetAttrValue(float32_t &attr_value) const
graphStatus GetAttrValue(AscendString &attr_value) const
graphStatus GetAttrValue(bool &attr_value) const
graphStatus GetAttrValue(Tensor &attr_value) const
graphStatus GetAttrValue(std::vector<int64_t> &attr_value) const
graphStatus GetAttrValue(std::vector<float32_t> &attr_value) const
graphStatus GetAttrValue(std::vector<AscendString> &attr_values) const
graphStatus GetAttrValue(std::vector<bool> &attr_value) const
graphStatus GetAttrValue(std::vector<Tensor> &attr_value) const
graphStatus GetAttrValue(std::vector<std::vector<int64_t>> &attr_value) const
graphStatus GetAttrValue(std::vector<ge::DataType> &attr_value) const
graphStatus GetAttrValue(ge::DataType &attr_value) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| attr_value | 输出 | 具体的属性取值 |

## 返回值说明

graphStatus类型：成功，返回`GRAPH_SUCCESS`，否则，返回`GRAPH_FAILED`。

## 异常处理

无

## 约束说明

无
