##### Expand（更改原 shape）

## 函数功能
对 shape 做补维，并将补维后的结果直接更新原 shape 对象。

## 函数原型
```cpp
ge::graphStatus Expand(Shape &shape) const
```

## 参数说明

| 参数   | 输入/输出 | 说明                               |
| ------ | --------- | ---------------------------------- |
| shape  | 输入/输出 | 需要进行补维操作的 shape 对象。 |

## 返回值说明
补维成功返回 `ge::GRAPH_SUCCESS`。

关于 `ge::graphStatus` 类型的定义，请参见 15.2.3.55 ge::graphStatus。

## 约束说明
无。

## 调用示例
```cpp
Shape shape({3, 256, 256}); // 设置原始shape 3,256,256
ExpandDimsType type1("1000");
auto ret = type1.Expand(shape); // ret = ge::GRAPH_SUCCESS, shape = 1,3,256,256
```
