##### MutableOriginShape

## 函数功能
获取可写的原始shape。

## 函数原型
```cpp
Shape &MutableOriginShape()
```

## 参数说明
无。

## 返回值说明
可写的原始shape。

## 约束说明
无。

## 调用示例
```cpp
StorageShape shape({3, 256, 256}, {256, 256, 3});
auto origin_shape = shape.MutableOriginShape(); // 3,256,256
```
