##### NeedAtomic

## 函数功能

获取是否需要 atomic clean 的标识值。

## 函数原型

```cpp
bool NeedAtomic() const
```

## 参数说明

无。

## 返回值说明

- `true`：代表需要做 atomic clean
- `false`：代表不需要做 atomic clean

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    bool is_need_atomic = context->NeedAtomic();
    // ...
}
```
