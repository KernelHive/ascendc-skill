##### GetOriginFormat

## 函数功能

获取 `CompileTimeTensorDesc` 所描述 Tensor 的原始数据排布格式。

## 函数原型

```cpp
ge::Format GetOriginFormat() const
```

## 参数说明

无。

## 返回值说明

`CompileTimeTensorDesc` 所描述 Tensor 的原始数据排布格式。

## 约束说明

无。

## 调用示例

```cpp
auto dtype_ = ge::DataType::DT_INT32;
StorageFormat fmt_(ge::Format::FORMAT_NC, ge::FORMAT_NCHW, {});
ExpandDimsType type_("1001");
gert::CompileTimeTensorDesc td;

td.SetDataType(dtype_);
auto dtype = td.GetDataType(); // ge::DataType::DT_INT32;

td.SetStorageFormat(fmt_.GetStorageFormat());
auto storage_fmt = td.GetStorageFormat(); // ge::FORMAT_NCHW

td.SetOriginFormat(fmt_.GetOriginFormat());
auto origin_fmt = td.GetOriginFormat(); // ge::Format::FORMAT_NC

td.SetExpandDimsType(type_);
auto type = td.GetExpandDimsType(); // type_("1001")
```
