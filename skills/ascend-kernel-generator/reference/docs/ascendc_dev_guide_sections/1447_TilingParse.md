##### TilingParse

## 函数功能

注册算子的 TilingParse 函数，用于解析算子编译阶段生成的算子信息 JSON 文件。在注册时需要注册算子自行指定数据类型 `T`，该数据类型用于保存解析后的算子信息。

用户需要为算子编写一个 `KernelFunc` 类型或者 `TilingParseFunc` 类型的函数，并使用下列对应的接口进行注册。

`KernelFunc` 类型定义如下：

```cpp
using KernelFunc = UINT32 (*)(KernelContext *context);
```

`TilingParseFunc` 类型定义如下：

```cpp
using TilingParseFunc = UINT32 (*)(TilingParseContext *context);
```

## 函数原型

```cpp
template<typename T>
OpImplRegisterV2 &TilingParse(KernelFunc const tiling_parse_func)
```

```cpp
template<typename T>
OpImplRegisterV2 &TilingParse(TilingParseFunc const tiling_parse_func)
```

## 参数说明

| 参数                | 输入/输出 | 说明                                                         |
|---------------------|-----------|--------------------------------------------------------------|
| `tiling_parse_func` | 输入      | 待注册的 TilingParse 函数，类型支持 2 种：`KernelFunc`、`TilingParseFunc`。 |

## 返回值说明

返回算子的 `OpImplRegisterV2` 对象，该对象新增注册了 TilingParse 函数。

## 约束说明

无。
