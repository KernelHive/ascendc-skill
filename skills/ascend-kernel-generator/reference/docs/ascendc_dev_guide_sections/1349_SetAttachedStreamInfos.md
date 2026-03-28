##### SetAttachedStreamInfos

## 函数功能

为当前算子附着一组 Stream 信息，用于指定该算子在哪个执行流（Stream）上运行。

## 函数原型

```cpp
ge::graphStatus SetAttachedStreamInfos(std::vector<StreamInfo> &stream_info_vec) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| stream_info_vec | 输入 | 执行流信息，std::vector<StreamInfo> 类型 |

## 返回值说明

返回 graphStatus 状态码。

## 约束说明

无

## 调用示例

```cpp
ge::graphStatus SetAttachedStreamInfos(ExeResGenerationContext* context) {
    std::vector<StreamInfo> stream_info_vec;
    auto status = context->SetAttachedStreamInfos(stream_info_vec);
    // ...
}
```
