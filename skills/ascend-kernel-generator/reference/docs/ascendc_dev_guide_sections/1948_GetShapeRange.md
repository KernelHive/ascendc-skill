##### GetShapeRange

## 函数功能
获取设置的shape变化范围。

## 函数原型
```cpp
graphStatus GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| range  | 输出      | 设置过的shape变化范围。 |

## 返回值
graphStatus类型：若成功，则该值为`GRAPH_SUCCESS`(即0)，其他值则为执行失败。

## 异常处理
无。

## 约束说明
无。
