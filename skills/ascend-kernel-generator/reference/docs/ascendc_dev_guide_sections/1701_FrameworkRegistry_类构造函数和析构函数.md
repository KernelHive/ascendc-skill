##### FrameworkRegistry 类构造函数和析构函数

## 函数功能

FrameworkRegistry 构造函数和析构函数。

## 函数原型

```cpp
FrameworkRegistry(const FrameworkRegistry &) = delete
FrameworkRegistry& operator = (const FrameworkRegistry &) = delete
~FrameworkRegistry()
static FrameworkRegistry& Instance()
```

## 参数说明

无

## 返回值

`Instance()` 返回 FrameworkRegistry 的单例对象。

## 异常处理

无

## 约束说明

无
