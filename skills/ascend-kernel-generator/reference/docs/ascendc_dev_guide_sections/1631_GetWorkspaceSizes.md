##### GetWorkspaceSizes

## 功能
获取workspace sizes指针，workspace大小以字节为单位。

## 函数原型
```c
size_t *GetWorkspaceSizes(const size_t workspace_count)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| workspace_count | 输入 | workspace的个数，取值不超过15.2.2.35.24 GetWorkspaceNum返回的workspace个数。超出时，会返回空指针。 |

## 返回值
workspace sizes指针。

## 约束
无。

## 调用示例
```c
ge::graphStatus Tiling4XXX(TilingContext* context) {
    auto ws = context->GetWorkspaceSizes(5);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // ...
}
```
