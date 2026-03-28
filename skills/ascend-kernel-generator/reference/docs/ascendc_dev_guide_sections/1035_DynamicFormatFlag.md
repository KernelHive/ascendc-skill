###### DynamicFormatFlag

## 功能说明

标识是否根据 `SetOpSelectFormat` 设置的函数自动推导算子输入输出支持的 dtype 和 format。设置为 `true`，则无需在原型注册时配置固定的 dtype 与 format，会调用推导函数来推导算子输入输出支持的 dtype 和 format。

## 函数原型

```cpp
OpAICoreConfig &DynamicFormatFlag(bool flag)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| flag | 输入 | 标记是否自动推导算子输入输出的 dtype 和 format。 |

## 返回值说明

`OpAICoreConfig` 类，请参考 15.1.6.1.7 `OpAICoreConfig`。

## 约束说明

无

## 调用示例

请参考 `SetOpSelectFormat` 节调用示例。
