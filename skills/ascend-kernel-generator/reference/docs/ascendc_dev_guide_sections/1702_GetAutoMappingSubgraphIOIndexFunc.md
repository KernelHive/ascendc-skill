##### GetAutoMappingSubgraphIOIndexFunc

## 函数功能

根据网络类型，获取已经注册的自动映射函数。

## 函数原型

```cpp
AutoMappingSubgraphIOIndexFunc GetAutoMappingSubgraphIOIndexFunc(domi::FrameworkType framework)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| framework | 输入 | 网络类型，FrameworkType类型定义请参考15.2.3.25.4 FrameworkType |

## 返回值

AutoMappingSubgraphIOIndexFunc：自动映射输入输出函数，函数类型详见 AutoMappingSubgraphIndex。

## 异常处理

无。

## 约束说明

无。
