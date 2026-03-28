##### GetStreamId

## 函数功能

返回当前算子绑定的 StreamId。

## 函数原型

```cpp
int64_t GetStreamId() const
```

## 参数说明

无

## 返回值说明

返回当前算子绑定的 StreamId。

## 约束说明

只有在每个算子只能绑定一个 Stream 的前提下可以使用该接口。

## 调用示例

```cpp
int64_t GetStreamId(ExeResGenerationContext* context) {
    auto streamId = context->GetStreamId();
    // ...
}
```
