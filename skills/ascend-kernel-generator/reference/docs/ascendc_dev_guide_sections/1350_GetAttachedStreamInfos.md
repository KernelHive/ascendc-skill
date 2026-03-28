##### GetAttachedStreamInfos

## 函数功能

返回当前算子绑定的所有 StreamInfo 列表。

## 函数原型

```cpp
std::vector<StreamInfo> GetAttachedStreamInfos() const
```

## 参数说明

无

## 返回值说明

返回当前算子绑定的所有 StreamInfo 列表。

## 约束说明

无

## 调用示例

```cpp
std::vector<StreamInfo> GetAttachedStreamInfos(ExeResGenerationContext* context) {
    std::vector<StreamInfo> streamInfoList = context->GetAttachedStreamInfos();
    // ...
}
```
