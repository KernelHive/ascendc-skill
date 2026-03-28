###### ParamType

## 功能说明

定义算子参数类型。

## 函数原型

```cpp
OpParamDef &ParamType(Option param_type)
```

## 参数说明

| 参数      | 输入/输出 | 说明                                                                 |
|-----------|-----------|----------------------------------------------------------------------|
| param_type | 输入      | 参数类型，Option 取值为：OPTIONAL（可选）、REQUIRED（必选）、DYNAMIC（动态输入）。 |

## 返回值说明

返回 OpParamDef 算子定义。OpParamDef 请参考 15.1.6.1.4 OpParamDef。

## 约束说明

无
