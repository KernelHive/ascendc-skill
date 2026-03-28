##### ParseParamsByOperatorFn

## 功能描述
注册解析用户自定义算子属性的函数。

## 函数原型
```cpp
OpRegistrationData &ParseParamsByOperatorFn(const ParseParamByOpFunc &parse_param_by_op_fn)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| `parse_param_by_op_fn` | 输入 | 解析用户自定义算子属性的函数，请参见回调函数 `ParseParamByOpFunc`。 |

## 回调函数 ParseParamByOpFunc
用户自定义并实现 `ParseParamByOpFunc` 类函数，完成原始模型中算子属性到适配昇腾AI处理器的模型中属性的映射，将结果填到 `Operator` 类中。

```cpp
Status ParseParamByOpFunc(const ge::Operator &op_origin, ge::Operator &op_dest)
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| `op_origin` | 输入 | 框架定义的 `Operator` 类对象，包含解析出的原始模型中自定义算子属性信息。关于 `Operator` 类，请参见 15.2.3.22 Operator。 |
| `op_dest` | 输出 | 适配昇腾AI处理器的模型中的算子数据结构，保存算子信息。关于 `Operator` 类，请参见 15.2.3.22 Operator。 |

## 约束说明
无。
