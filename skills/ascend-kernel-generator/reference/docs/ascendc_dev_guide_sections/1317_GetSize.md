##### GetSize

## 函数功能
获取当前存放的实际元素数量。

## 函数原型
```cpp
size_t GetSize() const
```

## 参数说明
无。

## 返回值说明
当前存放的实际元素数量。

## 约束说明
无。

## 调用示例
```cpp
// 创建ContinuousVectorVector对象cvv
...

// 增加元素
...
auto cv = cvv->add(inner_vector_capacity);
...

// 获取当前存放的实际元素数量
auto size = cvv->GetSize();
```
