##### GetCompileInfo

## 函数功能

获取编译信息 `CompileInfo`。

本方法用以获取算子 `TilingParse` 解析出来的编译信息，具体信息看具体算子对应的 `CompileInfo` 结构体。例如，对于 `Add` 算子来说，其编译信息如下：

```cpp
struct AddCompileInfo {
    int64_t a; // 输入a
    int64_t b; // 输入b
};
```

## 函数原型

```cpp
template<typename T> const T *GetCompileInfo() const
const void *GetCompileInfo() const
```

## 参数说明

| 参数 | 说明 |
|------|------|
| T    | 模板参数，CompileInfo 的类型。 |

## 返回值说明

- `template<typename T> const T *GetCompileInfo() const`：返回一个指定类型 `T` 的 `CompileInfo` 的指针。
- `const void *GetCompileInfo() const`：返回一个任意类型的 `CompileInfo` 的指针。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus TilingForAdd(TilingContext *context) {
    auto ci = context->GetCompileInfo<AddCompileInfo>();
    GE_ASSERT_NOTNULL(ci);
    ...
}
```
