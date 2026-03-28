##### Tiling

## 函数功能
注册算子的Tiling函数。

用户需要为算子编写一个 TilingKernelFunc 类型的函数，并使用该接口进行注册。同时可以指定 tiling 数据的最大长度，缺省值为 2048 字节。

TilingKernelFunc 类型定义如下：
```cpp
using TilingKernelFunc = UINT32 (*)(TilingContext *);
```

## 函数原型
```cpp
OpImplRegisterV2 &Tiling(TilingKernelFunc tiling_func, size_t max_tiling_data_size = 2048)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `tiling_func` | 输入 | 待注册的自定义 Tiling 函数，类型为 `TilingKernelFunc` |
| `max_tiling_data_size` | 输入 | tiling 数据的最大长度，默认值为 2048 字节 |

## 返回值说明
返回算子的 `OpImplRegisterV2` 对象，该对象新增注册了 Tiling 函数 `tiling_func`。

## 约束说明
无。
