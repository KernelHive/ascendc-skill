##### SetNeedIteration

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

标记Graph是否需要循环执行。

## 函数原型

```cpp
void SetNeedIteration(bool need_iteration)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| need_iteration | 输入 | 标记图是否需要循环执行。<br>取值：<br>● true：循环执行。<br>● false：不循环执行。 |

## 返回值说明

无

## 约束说明

需要与 `npu_runconfig/iterations_per_loop`、`npu_runconfig/loop_cond`、`npu_runconfig/one`、`npu_runconfig/zero` 等搭配使用，用户需要先构造带有 `npu_runconfig/iterations_per_loop`、`npu_runconfig/loop_cond`、`npu_runconfig/one`、`npu_runconfig/zero` 名字的 variable 算子节点。
