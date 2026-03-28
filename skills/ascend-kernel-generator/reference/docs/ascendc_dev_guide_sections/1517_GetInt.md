##### GetInt

## 函数功能
获取int类型的属性值。

## 函数原型
```cpp
const int64_t *GetInt(const size_t index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| index | 输入 | 属性在IR原型定义中以及在OP_IMPL注册中的索引。 |

## 返回值说明
指向属性值的指针。

## 约束说明
无。

## 调用示例
```cpp
const RuntimeAttrs * runtime_attrs = kernel_context->GetAttrs();
const int64_t *attr0 = runtime_attrs->GetInt(0);
```
