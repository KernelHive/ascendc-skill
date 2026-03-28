##### FusionParseParamsFn（Overload）

## 函数功能

注册解析融合算子属性的函数，为 `FusionParseParamsFn` 的重载函数。

## 函数原型

```cpp
OpRegistrationData &FusionParseParamsFn(const FusionParseParamByOpFunc &fusion_parse_param_fn)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| `fusion_parse_param_fn` | 输入 | 解析融合算子属性的函数，请参见回调函数 `FusionParseParamByOpFunc`。 |

## 回调函数 FusionParseParamByOpFunc

用户自定义并实现 `FusionParseParamByOpFunc` 类函数，完成原始模型中属性到适配昇腾AI处理器的模型中的属性映射，将结果填到 `Operator` 类中。

```cpp
Status FusionParseParamByOpFunc(const std::vector<ge::Operator> &op_src, ge::Operator &op_dest)
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| `op_src` | 输入 | 一组 scope 内存储原始模型中算子属性的融合算子数据结构。<br>关于 `Operator` 类，请参见 15.2.3.22 Operator。 |
| `op_dest` | 输出 | 融合算子数据结构，保存融合算子信息。<br>关于 `Operator` 类，请参见 15.2.3.22 Operator。 |

## 调用示例

```cpp
REGISTER_CUSTOM_OP(XXXXXX)
    .FrameworkType(TENSORFLOW)
    .FusionParseParamsFn(FusionParseParamsFn)
    .OriginOpType(XXXXX)
    .ImplyType(XXXXX);
```
