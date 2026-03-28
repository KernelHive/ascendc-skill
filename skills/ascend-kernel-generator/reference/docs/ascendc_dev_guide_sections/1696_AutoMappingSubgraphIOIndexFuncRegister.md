#### AutoMappingSubgraphIOIndexFuncRegister

## 函数功能

FrameworkRegistry类的封装，通过类的构造函数调用FrameworkRegistry类的`AddAutoMappingSubgraphIOIndexFunc`函数完成映射函数的注册。

## 函数原型

```cpp
AutoMappingSubgraphIOIndexFuncRegister(domi::FrameworkType framework, AutoMappingSubgraphIOIndexFunc fun)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| framework | 输入 | 网络类型，FrameworkType类型定义请参考15.2.3.25.4 FrameworkType |
| fun | 输入 | 自动映射输入输出函数，函数类型详见AutoMappingSubgraphIndex |

## 返回值

无

## 异常处理

无

## 约束说明

无
