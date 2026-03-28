##### Expand（输出新 shape）

## 函数功能

对 shape 做补维，并将补维后的结果写入指定的输出 shape 对象。

## 函数原型

```cpp
ge::graphStatus Expand(const Shape &shape, Shape &out_shape) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| shape | 输入 | 输入 shape，补维前 shape |
| out_shape | 输出 | 输出 shape，补维后 shape |

## 返回值说明

补维成功返回 `ge::GRAPH_SUCCESS`；失败则返回 `ge::GRAPH_FAILED`。

关于 `ge::graphStatus` 类型的定义，请参见 15.2.3.55 ge::graphStatus。

## 约束说明

无。

## 调用示例

```cpp
Shape origin_shape({3, 256, 256}); // 设置原始 shape 3x256x256
Shape out_shape;
ExpandDimsType type1("1000");
ExpandDimsType type2("10000");
ExpandDimsType type3("1001");

auto ret = type1.Expand(origin_shape, out_shape); // ret = ge::GRAPH_SUCCESS, out_shape = 1,3,256,256
ret = type2.Expand(origin_shape, out_shape); // ret = ge::GRAPH_FAILED
ret = type3.Expand(origin_shape, out_shape); // ret = ge::GRAPH_SUCCESS, out_shape = 1,3,256,1,256
```
