##### GetDynamicInputDataType

## 函数功能

根据算子原型定义中的输入索引获取对应动态输入的数据类型。

## 函数原型

```cpp
ge::DataType GetDynamicInputDataType(const size_t ir_index, const size_t relative_index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | 动态输入在算子IR原型定义中的索引，从0开始计数 |
| relative_index | 输入 | 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的取值范围是[0,2] |

## 返回值说明

返回指定输入的数据类型，若输入的ir_index或者relative_index非法，返回DT_UNDEFINED。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus InferDataTypeForXXX(InferDataTypeContext *context) {
    auto data_type = context->GetDynamicInputDataType(1, 2);
    // ...
}
```
