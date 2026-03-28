###### DynamicRankSupportFlag

## 功能说明
标识算子是否支持 dynamic rank（动态维度）。

## 函数原型
```cpp
OpAICoreConfig &DynamicRankSupportFlag(bool flag)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| flag | 输入 | ● true：表示算子支持 dynamic rank，算子支持 shape 包含（-2），用于判断是否进行动态编译；<br>● false：表示算子不支持 dynamic rank。 |

## 返回值说明
返回 OpAICoreConfig 类，请参考 15.1.6.1.7 OpAICoreConfig。

## 约束说明
无
