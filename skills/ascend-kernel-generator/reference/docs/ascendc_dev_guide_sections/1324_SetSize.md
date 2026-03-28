##### SetSize

## 函数功能
设置当前保存的元素个数。

## 函数原型
```cpp
ge::graphStatus SetSize(const size_t size)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| size | 输入 | 当前保存的元素个数 |

## 返回值说明
- 成功时返回 `ge::GRAPH_SUCCESS`
- 设置的 `size > capacity` 时，返回失败 `ge::GRAPH_FAILED`

## 约束说明
无

## 调用示例
```cpp
size_t capacity = 100U;
auto cv_holder = ContinuousVector::Create<int64_t>(capacity);
auto cv = reinterpret_cast<ContinuousVector *>(cv_holder.get());
auto ret = cv->SetSize(10U);  // ge::GRAPH_SUCCESS
ret = cv->GetSize(101U);      // ge::GRAPH_FAILED
```
