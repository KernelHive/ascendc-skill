##### SetScheduleMode

## 功能
设置算子在NPU上执行时的调度模式。

## 函数原型
```cpp
ge::graphStatus SetScheduleMode(const uint32_t schedule_mode)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| schedule_mode | 输入 | 0：普通模式，默认情况下为普通模式。<br>1：batchmode模式，核间同步算子需要设置该模式。 |

## 返回值
- 设置成功时返回 `ge::GRAPH_SUCCESS`
- 设置失败时返回 `ge::GRAPH_FAILED`

关于 `graphStatus` 的定义，请参见 15.2.3.55 ge::graphStatus。

## 约束说明
无。

## 调用示例
```cpp
ge::graphStatus TilingForAdd(TilingContext *context) {
    uint32_t batch_mode = 1U;
    auto ret = context->SetScheduleMode(batch_mode);
    GE_ASSERT_SUCCESS(ret);
    // ...
}
```
