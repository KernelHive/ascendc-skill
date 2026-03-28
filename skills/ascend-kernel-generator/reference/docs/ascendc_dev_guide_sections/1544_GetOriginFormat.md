##### GetOriginFormat

## 函数功能
获取原始 format。

## 函数原型
```cpp
ge::Format GetOriginFormat() const
```

## 参数说明
无。

## 返回值说明
原始 format。

## 约束说明
无。

## 调用示例
```cpp
ExpandDimsType dim_type("1100");
StorageFormat format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_C1HWNC0, dim_type);
auto origin_format = format.GetOriginFormat(); // Format::FORMAT_NCHW
```
