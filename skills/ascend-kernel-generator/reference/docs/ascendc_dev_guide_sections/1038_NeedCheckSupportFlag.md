###### NeedCheckSupportFlag

## 功能说明

标识是否在算子融合阶段调用算子参数校验函数进行 data type 与 shape 的校验。

- 若配置为 `"true"`，框架会调用通过 `SetCheckSupport` 设置的算子参数校验函数，检查算子是否支持指定输入，此场景下需要自行实现算子参数校验的回调函数。
- 若配置为 `"false"`，表示不需要进行校验。

## 函数原型

```cpp
OpAICoreConfig &NeedCheckSupportFlag(bool flag)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| flag | 输入 | 标识是否在算子融合阶段调用算子参数校验函数进行 data type 与 shape 的校验。 |

## 返回值说明

`OpAICoreConfig` 类，请参考 15.1.6.1.7 OpAICoreConfig。

## 约束说明

无

## 调用示例

请参考 `SetCheckSupport` 节调用示例。
