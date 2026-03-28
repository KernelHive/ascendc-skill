###### SetTiling

## 功能说明

注册Tiling函数。Tiling函数的原型是固定的，接受一个TilingContext作为输入，在此context上可以获取到输入、输出的Shape指针等内容。注册的Tiling函数由框架调用，调用时会传入TilingContext参数。

## 函数原型

```cpp
OpAICoreDef &SetTiling(gert::OpImplRegisterV2::TilingKernelFunc func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| func | 输入 | Tiling函数。TilingKernelFunc类型定义如下：<br>`using TilingKernelFunc = UINT32 (*)(TilingContext *);` |

## 返回值说明

OpAICoreDef算子定义，OpAICoreDef请参考15.1.6.1.6 OpAICoreDef。

## 约束说明

无
