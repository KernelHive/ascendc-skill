#### BROADCAST_INFER

## 函数功能

提供公共函数宏封装，供算子开发者开发 InferShape 函数。该函数基于 2 个输入的 shape，设置输出的 shape。该宏只是设置 shape，未设置 dtype。

- 如果 2 个输入的 shape 一致，会按输入的 shape 设置输出 shape。
- 如果 2 个输入的 shape 不一致，会按照 Broadcast 的策略，取 2 个输入 shape 的并集。

例如：输入 shape 分别为 `(1,2,3,4)` 和 `(3,1,3,4)`，则该宏会设置算子的输出 shape 为 `(3,2,3,4)`。

## 函数原型

```cpp
BROADCAST_INFER(in1_name, in2_name, out_name)
```

该函数会自动调用如下函数：

```cpp
graphStatus BroadCastInfer(
    const function<vector<int64_t>()> &get_in1_shape,
    const function<vector<int64_t>()> &get_in2_shape,
    const function<void(const std::vector<int64_t> &y_shape)> &set_out_shape
);
```

## 约束说明

无。

## 参数说明

| 参数名    | 输入/输出 | 描述           |
|-----------|-----------|----------------|
| in1_name  | 输入      | 算子第一个输入 |
| in2_name  | 输入      | 算子第二个输入 |
| out_name  | 输入      | 算子输出       |

## 返回值

执行成功或失败。

## 调用示例

```cpp
IMPLEMT_INFERFUNC(RightShift, RightShiftInfer) {
    DataType type = op.GetInputDesc("x").GetDataType();
    SET_OUTPUT_TYPE(op, "z", type);
    return BROADCAST_INFER("x", "y", "z")(op);
}
```
