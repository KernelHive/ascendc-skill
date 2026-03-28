###### IgnoreContiguous

## 功能说明

某些算子支持非连续的 tensor，在算子的实现中对非连续的 tensor 做了转换处理。配置该参数后，框架会忽略对非连续的校验。

## 函数原型

```cpp
OpParamDef &IgnoreContiguous(void)
```

## 参数说明

无

## 返回值说明

OpParamDef 算子定义，OpParamDef 请参考 15.1.6.1.4 OpParamDef。

## 约束说明

无
