##### GetAttrPointer

## 函数功能
获取指定索引的算子属性，返回指向此属性的指针。

## 函数原型
```cpp
template<typename T> const T *GetAttrPointer(size_t index) const
```

## 参数说明

| 参数  | 输入/输出 | 说明 |
|-------|-----------|------|
| T     | 输入      | 属性类型 |
| index | 输入      | 属性在IR原型定义中的索引 |

## 返回值说明
指向属性的指针。

## 约束说明
无。

## 调用示例
```cpp
const RuntimeAttrs * runtime_attrs = kernel_context->GetAttrs();
const gert::ContinuousVector attr0 = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(0);
```
