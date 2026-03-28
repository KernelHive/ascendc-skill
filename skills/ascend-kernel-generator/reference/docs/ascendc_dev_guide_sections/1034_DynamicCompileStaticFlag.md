###### DynamicCompileStaticFlag

## 功能说明

用于标识该算子实现是否支持入图时的静态 Shape 编译。

## 函数原型

```cpp
OpAICoreConfig &DynamicCompileStaticFlag(bool flag)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| flag | 输入 | 用户开发的自定义算子，如果需要支持入图时静态 Shape 场景下的编译，需要配置该选项为 `true`，否则配置为 `false`。 |

## 返回值说明

`OpAICoreConfig` 类，请参考 15.1.6.1.7 OpAICoreConfig。

## 约束说明

无
