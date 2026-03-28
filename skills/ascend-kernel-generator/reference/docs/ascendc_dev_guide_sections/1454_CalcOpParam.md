##### CalcOpParam

## 函数功能

该接口用于 GenerateTask 阶段的资源计算和申请。

## 函数原型

```cpp
OpImplRegisterV2 &CalcOpParam(OpCalcParamKernelFunc calc_op_param_func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `calc_op_param_func` | 输入 | 待注册的 OpCalcParamKernel 函数。<br>`OpCalcParamKernelFunc` 类型定义如下：<br>`using OpCalcParamKernelFunc = UINT32 (*)(ExeResGenerationContext* context);` |

## 返回值说明

返回算子的 `OpImplRegisterV2` 对象本身，该对象新增注册了 `OpCalcParamKernelFunc` 函数。

## 约束说明

无
