##### GetExecuteMode

## 函数功能
返回当前算子运行的执行模式。

## 函数原型
```cpp
ExecuteMode GetExecuteMode() const
```

## 参数说明
无

## 返回值说明
返回 `ExecuteMode` 枚举值，表示当前算子的运行模式。

```cpp
enum class ExecuteMode {
    kStaticOffloadExecute, // 静态图模式执行
    kDynamicExecute,       // 动态图模式执行
    kEnd                   // end
};
```

## 约束说明
无

## 调用示例
```cpp
ExecuteMode GetExecuteMode(ExeResGenerationContext *context) {
    ExecuteMode mode = *context->GetExecuteMode();
    // ...
}
```
