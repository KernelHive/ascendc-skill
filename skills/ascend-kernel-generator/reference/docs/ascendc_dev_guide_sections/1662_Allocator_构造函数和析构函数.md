##### Allocator 构造函数和析构函数

## 函数功能
Allocator 构造函数和析构函数。

## 函数原型
```cpp
Allocator() = default
virtual ~Allocator() = default
Allocator(const Allocator &) = delete
Allocator &operator=(const Allocator &) = delete
```

## 参数说明
无。

## 返回值
无。

## 异常处理
无。

## 约束说明
纯虚类需要用户派生。
