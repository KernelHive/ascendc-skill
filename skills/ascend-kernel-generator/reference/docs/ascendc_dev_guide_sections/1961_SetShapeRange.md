##### SetShapeRange

## 函数功能
设置 shape 的变化范围。

## 函数原型
```cpp
graphStatus SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| range  | 输入      | shape 代表的变化范围。vector 中的每一个元素为一个 pair，pair 的第一个值为该维度上的 dim 最小值，第二个值为该维度上 dim 的最大值。 |

**示例**：
该 tensor 的 shape 为 `{1, 1, -1, 2}`，第三个轴的最大值为 100，则 range 可设置为 `{{1, 1}, {1, 1}, {1, 100}, {2, 2}}`。

## 返回值
graphStatus 类型：若成功，则该值为 `GRAPH_SUCCESS`（即 0），其他值则为执行失败。

## 异常处理
无。

## 约束说明
无。
