###### Workspace

## 功能说明
传入指向 `gert::ContinuousVector` 的指针。

## 函数原型
```cpp
ContextBuilder &Workspace(gert::ContinuousVector *workspace)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| workspace | 输入 | 指向 `gert::ContinuousVector` 类的 `void*` 指针 |

## 返回值说明
返回当前 `ContextBuilder` 的对象。

## 约束说明
由于 `TilingContext` 与 `KernelContext`、`TilingParseContext` 内部数据排序不同，`Workspace()` 只支持以调用 `BuildTilingContext()` 为前提来使用；其他场景建议用 `Outputs` 接口，否则发生未定义行为。

## 调用示例
```cpp
void AddWorkspaceData(gert::ContinuousVector *ws)
{
    // ...
    auto builder = context_ascendc::ContextBuilder()
        .Workspace(ws)
        .BuildTilingContext();
    // ...
}
```
