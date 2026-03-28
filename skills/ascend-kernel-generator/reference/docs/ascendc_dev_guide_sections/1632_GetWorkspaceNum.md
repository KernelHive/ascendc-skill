##### GetWorkspaceNum

## 函数功能
获取 workspace 个数。

## 函数原型
```cpp
size_t GetWorkspaceNum() const
```

## 参数说明
无。

## 返回值说明
workspace 的个数。

## 约束说明
无。

## 调用示例
```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    auto ws_num = context->GetWorkspaceNum();
    // ...
}
```
