##### GetSize

## 函数功能
获取当前保存的元素个数。

## 函数原型
```cpp
size_t GetSize() const
```

## 参数说明
无。

## 返回值说明
当前保存的元素个数。

## 约束说明
无。

## 调用示例
```cpp
size_t capacity = 100U;
auto cv_holder = ContinuousVector::Create<int64_t>(capacity);
auto cv = reinterpret_cast<ContinuousVector *>(cv_holder.get());
auto size = cv->GetSize(); // 0U
```
