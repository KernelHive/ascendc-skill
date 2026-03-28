##### GetCapacity

## 功能
获取最大可保存的元素个数。

## 函数原型
```cpp
size_t GetCapacity() const
```

## 参数
无。

## 返回值
最大可保存的元素个数。

## 约束
无。

## 调用示例
```cpp
size_t capacity = 100U;
auto cv_holder = ContinuousVector::Create<int64_t>(capacity);
auto cv = reinterpret_cast<ContinuousVector *>(cv_holder.get());
auto cap = cv->GetCapacity(); // 100U
```
