##### GetData

## 函数功能

获取首个元素的指针地址，`[GetData(), reinterpret_cast<T *>(GetData()) + GetSize())` 中的数据即为当前容器中保存的数据。

## 函数原型

```cpp
const T *GetData() const
```

## 参数说明

无。

## 返回值说明

首个元素的指针地址。

## 约束说明

无。

## 调用示例

```cpp
size_t capacity = 100U;
auto cv_holder = ContinuousVector::Create<int64_t>(capacity);
auto cv = reinterpret_cast<TypedContinuousVector *>(cv_holder.get());
auto cap = cv->GetData<int64_t>();
```
