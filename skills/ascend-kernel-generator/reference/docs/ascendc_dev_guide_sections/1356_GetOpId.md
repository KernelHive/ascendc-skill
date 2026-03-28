##### GetOpId

## 函数功能
获取当前算子的唯一标识ID。

## 函数原型
```cpp
int64_t GetOpId() const
```

## 参数说明
无

## 返回值说明
返回一个 `int64_t` 类型的整数，代表这个算子在计算图中的唯一编号。

## 约束说明
无

## 调用示例
```cpp
int64_t GetOpId(ExeResGenerationContext* context) {
    auto opId = context->GetOpId();
    // ...
}
```
