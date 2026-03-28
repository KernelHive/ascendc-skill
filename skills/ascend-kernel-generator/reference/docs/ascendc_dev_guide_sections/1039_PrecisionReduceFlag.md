###### PrecisionReduceFlag

## 功能说明

此字段用于进行 ATC 模型转换或者进行网络调测时，控制算子的精度模式。只有当精度模式（`precision_mode`）配置为混合精度（`allow_mix_precision`）前提下生效。

## 函数原型

```cpp
OpAICoreConfig& PrecisionReduceFlag(bool flag)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| flag | 输入 | ● 若配置为 `false`，则认为是黑名单，算子必须保持算子本身的原始数据类型。<br>● 若配置为 `true`，则认为是白名单，如果算子既支持 float32 又支持 float16 数据类型，同时算子的原图格式是 float32 或者 float16 的情况下，优先为算子选择 float16 数据类型。<br>● 若未配置这个字段，则认为是灰名单，在有上一个算子的情况下，选择和上一个算子相同的数据类型，否则选择当前算子的原始数据类型。 |

## 返回值说明

`OpAICoreConfig` 类，请参考 15.1.6.1.7 OpAICoreConfig。

## 约束说明

无
