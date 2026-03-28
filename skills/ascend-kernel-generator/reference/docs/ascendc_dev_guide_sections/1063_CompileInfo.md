###### CompileInfo

## 功能说明
将指向 CompileInfo 的指针传入 TilingContext。

## 函数原型
```cpp
ContextBuilder &CompileInfo(void *compileInfo)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| compileInfo | 输入 | 指向 CompileInfo 的 void 指针 |

## 返回值说明
返回当前 ContextBuilder 的对象。

## 约束说明
由于 TilingContext 与 KernelContext、TilingParseContext 内部数据排序不同，CompileInfo() 只支持以调用 BuildTilingContext() 为前提来使用；其他场景建议用 Outputs 接口，否则发生未定义行为。

## 调用示例
```cpp
void AddCompileInfo(TilingParseContext *tilingParseContext)
{
    void *compileInfo = *tilingParseContext->GetOutputPointer<void **>(0);
    auto kernelContextHolder = context_ascendc::ContextBuilder()
        // 增加算子输入输出接口的调用
        .CompileInfo(compileInfo)
        .BuildTilingContext();
}
```
