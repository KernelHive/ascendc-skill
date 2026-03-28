##### AppendAttr

## 函数功能

追加算子IR原型的属性信息，下标从0开始，用于构造各子类Context的基类 `ExtendedKernelContext` 中的 `ExtendedInfo` 信息。

构造完成后，通过Context的基类接口 `GetAttr` 获取到的 `RuntimeAttrs` 中属性的顺序与构造时追加的顺序一致。

## 示例

```cpp
bool attr0 = true;
int64_t attr1 = 1;
vector<int64_t> attr2 = {1, 2, 3, 4};
context_builder.AppendAttr(attr0).AppendAttr(attr1).AppendAttr(attr2);
```

Build构造完成后结果如下：

- `ctx->GetAttrs()->GetBool(0)` → `attr0`
- `ctx->GetAttrs()->GetInt(1)` → `attr1`
- `ctx->GetAttrs()->GetListInt(2)` → `attr2`

## 函数原型

```cpp
T &AppendAttr(bool attr)
T &AppendAttr(int64_t attr)
T &AppendAttr(float attr)
T &AppendAttr(const ge::AscendString &attr)
T &AppendAttr(const std::vector<bool> &attr)
T &AppendAttr(const std::vector<int64_t> &attr)
T &AppendAttr(const std::vector<float> &attr)
T &AppendAttr(const std::vector<ge::AscendString> &attr)
T &AppendAttr(const std::vector<std::vector<int64_t>> &attr)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| attr | 输入 | 属性值，当前仅支持以下类型：<br>• `bool`<br>• `int64_t`<br>• `float`<br>• `ge::AscendString`<br>• `std::vector<bool>`<br>• `std::vector<int64_t>`<br>• `std::vector<float>`<br>• `std::vector<ge::AscendString>`<br>• `std::vector<std::vector<int64_t>>` |

## 返回值说明

返回子类对象 `T` 类型的引用，用于支持子类链式调用。

## 约束说明

无。
