##### ParseParamsFn

## 函数功能
注册解析算子属性的函数。

## 函数原型
```cpp
OpRegistrationData &ParseParamsFn(const ParseParamFunc &parseParamFn)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| parseParamFn | 输入 | 解析算子属性的函数 ParseParamFunc，请参见回调函数 ParseParamFunc。<br>针对 TensorFlow 框架，若原始 TensorFlow 框架算子属性与适配昇腾 AI 处理器的模型中算子属性一一对应（属性个数与顺序一致），可直接使用 AutoMappingFn 函数自动实现映射。 |

## 约束说明
对于自定义算子插件，ParseParamsFn 后续版本将会废弃，请使用 15.2.3.25.7 ParseParamsByOperatorFn 接口进行算子属性的解析。

若用户已使用 ParseParamsFn 接口进行了算子插件的开发，请执行如下操作进行新接口适配：

1. 请重新使用 15.2.3.25.7 ParseParamsByOperatorFn 接口进行算子插件的开发。
2. 请基于新版本自定义算子样例工程的编译脚本重新进行自定义算子工程的编译。

## 回调函数 ParseParamFunc
用户自定义并实现 FusionParseParamFunc 类函数，完成原始模型中算子属性到适配昇腾 AI 处理器的模型中算子属性映射，将结果填到 Operator 类中。

```cpp
using ParseParamFunc = std::function<domi::Status(const google::protobuf::Message *, ge::Operator &)>
```

- 第一个参数为输入，protobuf 格式的数据结构（来源于原始模型的 prototxt 文件），包含算子属性信息。
- 第二个参数为输出，适配昇腾 AI 处理器的模型的算子数据结构，保存算子信息。

关于 Operator 类，请参见 15.2.3.22 Operator。
