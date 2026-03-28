##### AutoMappingFnDynamic

## 函数功能

动态输入/输出算子的自动映射回调函数。

## 函数原型

```cpp
Status AutoMappingFnDynamic(const google::protobuf::Message *op_src, ge::Operator &op,
std::map<std::string, std::pair<std::string, std::string>> dynamic_name_attr_value, int32_t in_pos = -1, int32_t
out_pos = -1)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| op_src | 输入 | 转换前原始模型中的算子，包含原始模型中算子的属性 |
| op | 输入 | 适配昇腾AI处理器的算子 |
| dynamic_name_attr_value | 输入 | 描述动态输入输出实际个数，key表示动态端口是输入还是输出 |
| in_pos | 输入 | 动态输入的端口id |
| out_pos | 输入 | 动态输出的端口id |

**dynamic_name_attr_value参数说明：**
- key的取值：
  - `in`：代表算子的输入
  - `out`：代表算子的输出

## 约束说明

若原始TensorFlow算子与适配昇腾AI处理器的算子属性无法一一映射，AutoMappingFnDynamic函数无法应用于回调函数ParseParamsByOperatorFn中。

此种场景下，请在回调函数ParseParamsByOperatorFn中使用15.2.3.25.28 AutoMappingByOpFnDynamic接口进行可以映射成功的属性的自动解析，使用示例请参见调用示例。

## 代码示例

### 动态输入的代码示例

```cpp
// register MapStage op to GE
Status MapStageMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
    map<string, pair<string, string>> value;
    value["in"] = pair<string, string>("values", "fake_dtypes");
    AutoMappingFnDynamic(op_src, op, value);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("MapStage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapStage")
    .ParseParamsFn(MapStageMapping)
    .ImplyType(ImplyType::AI_CPU);
```

### 动态输出的代码示例

```cpp
Status AutoMappingFnSplit(const google::protobuf::Message* op_src, ge::Operator& op) {
    map<string, pair<string, string>> value;
    value["out"] = pair<string, string>("y", "num_split");
    AutoMappingFnDynamic(op_src, op, value);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Split")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Split")
    .ParseParamsFn(AutoMappingFnSplit)
    .ImplyType(ImplyType::TVM);
```
