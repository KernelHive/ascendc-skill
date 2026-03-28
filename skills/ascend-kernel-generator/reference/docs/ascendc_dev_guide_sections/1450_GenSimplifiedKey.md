##### GenSimplifiedKey

## 函数功能

注册算子的 `GenSimplifiedKey` 函数，以提供一个更加快速的二进制匹配 key 值。

用户需要为算子编写一个 `GenSimplifiedKey` 类型的函数，并使用该接口进行注册。

`GenSimplifiedKey` 类型定义如下：

```cpp
using GenSimplifiedKeyKernelFunc = UINT32 (*)(TilingContext *, ge::char_t *);
```

`GenSimplifiedKey` 函数接受一个 `TilingContext` 类型参数和 `ge::char_t` 类型参数作为输入，通过该输入算子可自定义 simplified key 生成逻辑。

## 函数原型

```cpp
OpImplRegisterV2 &GenSimplifiedKey(GenSimplifiedKeyKernelFunc gen_simplifiedkey_func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `gen_simplifiedkey_func` | 输入 | 要注册的自定义 `GenSimplifiedKey` 函数，类型为 `GenSimplifiedKeyKernelFunc`。<br>`GenSimplifiedKeyKernelFunc` 类型定义如下：<br>`using GenSimplifiedKeyKernelFunc = UINT32 (*)(TilingContext *, ge::char_t *);` |

## 返回值说明

返回算子的 `OpImplRegisterV2` 对象，该对象新增注册了生成二进制简化匹配 key 函数。

## 约束说明

无。
