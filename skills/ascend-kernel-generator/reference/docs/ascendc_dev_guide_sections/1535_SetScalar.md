##### SetScalar

## 函数功能
设置 shape 为标量。

## 函数原型
```cpp
void SetScalar()
```

## 参数说明
无。

## 返回值说明
无。

## 约束说明
无。

## 调用示例
```cpp
Shape shape0({3, 256, 256});
shape0.IsScalar(); // false
shape0.SetScalar();
shape0.IsScalar(); // true
```
