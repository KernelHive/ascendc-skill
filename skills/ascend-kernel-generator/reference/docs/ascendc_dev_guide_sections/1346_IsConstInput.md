##### IsConstInput

## 函数功能

用于判断某个输入张量是否为常量数据。

## 函数原型

```cpp
bool IsConstInput(const ge::AscendString &name) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| name | 输入 | 输入张量的名称 |

## 返回值说明

- `true`：表示该输入是常量
- `false`：表示是动态输入（如输入数据、中间结果等）

## 约束说明

无

## 调用示例

```cpp
bool IsConstInput(ExeResGenerationContext *context) {
    ge::AscendString name = "XXX";
    bool status = *context->IsConstInput(name);
    // ...
}
```
