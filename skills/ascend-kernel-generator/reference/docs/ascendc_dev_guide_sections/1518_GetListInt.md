##### GetListInt

## 函数功能
获取 list int 类型的属性值。

## 函数原型
```cpp
const TypedContinuousVector<int64_t> *GetListInt(const size_t index) const
```

## 参数说明

| 参数  | 输入/输出 | 说明 |
|-------|-----------|------|
| index | 输入      | 属性在 IR 原型定义中以及在 OP_IMPL 注册中的索引。 |

## 返回值说明
指向属性值的指针。

关于 `TypedContinuousVector` 类型的定义，请参见 15.2.2.37 `TypedContinuousVector`。

## 约束说明
无。

## 调用示例
```cpp
const RuntimeAttrs *runtime_attrs = kernel_context->GetAttrs();
const TypedContinuousVector<int64_t> *attr0 = runtime_attrs->GetListInt(0);
```
