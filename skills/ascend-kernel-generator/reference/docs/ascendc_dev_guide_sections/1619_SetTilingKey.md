##### SetTilingKey

## 函数功能

设置 TilingKey。

不同的 kernel 实现分支可以通过 TilingKey 来标识，host 侧设置 TilingKey 后，可以选择对应的分支。例如，一个算子在不同的 shape 下，有不同的算法逻辑，kernel 侧可以通过 TilingKey 来选择不同的算法逻辑，在 host 侧 Tiling 算法也有差异，host/kernel 侧通过相同的 TilingKey 进行关联。

## 函数原型

```cpp
ge::graphStatus SetTilingKey(const uint64_t tiling_key)
```

## 参数说明

| 参数       | 输入/输出 | 说明                     |
|------------|-----------|--------------------------|
| tiling_key | 输入      | 需要设置的 tiling key。 |

## 返回值说明

成功时返回 `ge::GRAPH_SUCCESS`。

关于 graphStatus 定义，请参见 15.2.3.55 ge::graphStatus。

## 约束说明

tiling_key 的取值范围在 uint64_t 数据类型范围内，但不可以取值为 `UINT64_MAX`。

## 调用示例

```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    auto ret = context->SetTilingKey(20);
    // ...
}
```
