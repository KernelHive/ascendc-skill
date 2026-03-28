##### OpExecuteFunc

## 函数功能

在单个算子包含多 kernel 组合执行逻辑的场景下，算子可以通过该接口设置算子级的回调函数，回调函数内实现多 kernel 的下发。该功能为预留特性，暂不支持。

## 函数原型

```cpp
OpImplRegisterV2 &OpExecuteFunc(OpExecFunc op_execute_func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `op_execute_func` | 输入 | 注册的自定义 OpExecuteFunc 函数，类型为 `OpExecFunc`。<br>`OpExecFunc` 类型定义如下：<br>`using OpExecFunc = UINT32 (*)(OpExecuteContext *);` |

## 返回值说明

返回算子的 `OpImplRegisterV2` 对象，该对象新增注册了算子级的回调函数。

## 约束说明

无。
