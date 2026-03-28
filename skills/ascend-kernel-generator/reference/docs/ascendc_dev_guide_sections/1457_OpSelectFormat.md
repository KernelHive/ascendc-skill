##### OpSelectFormat

## 函数功能

注册一个格式选择函数，获取数据类型和数据格式，由算子自行决定支持情况。

## 函数原型

```cpp
OpImplRegisterV2 &OpSelectFormat(OP_CHECK_FUNC_V2 op_select_format_func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| select_format_func | 输入 | 待注册的 OP_CHECK_FUNC_V2 函数 |

`OP_CHECK_FUNC_V2` 类型定义如下：

```cpp
using OP_CHECK_FUNC_V2 = ge::graphStatus (*)(
    const OpCheckContext *context, 
    ge::AscendString &result
);
```

## 返回值说明

返回算子的 `OpImplRegisterV2` 对象本身，该对象新增注册 `OP_CHECK_FUNC_V2` 函数。

## 约束说明

无
