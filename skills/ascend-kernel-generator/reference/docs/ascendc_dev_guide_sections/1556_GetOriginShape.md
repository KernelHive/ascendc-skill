##### GetOriginShape

## 函数功能
获取原始 shape。

## 函数原型
```cpp
const Shape &GetOriginShape() const
```

## 参数说明
无。

## 返回值说明
原始 shape。

## 约束说明
无。

## 调用示例
```cpp
StorageShape shape({3, 256, 256}, {256, 256, 3});
auto origin_shape = shape.GetOriginShape(); // 3,256,256
```
