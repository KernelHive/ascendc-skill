##### GenerateTask

## 函数功能

GenerateTask 阶段具体 Task 的生成和处理。

## 函数原型

```cpp
OpImplRegisterV2 &GenerateTask(OpGenTaskKernelFunc gen_task_func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| gen_task_func | 输入 | 待注册的 OpGenTaskKernelFunc 函数 |

OpGenTaskKernelFunc 类型定义如下：

```cpp
using OpGenTaskKernelFunc = UINT32 (*)(const ExeResGenerationContext *context,
                                       std::vector<std::vector<uint8_t>> &tasks);
```

## 返回值说明

返回算子的 OpImplRegisterV2 对象本身，该对象新增注册 OpGenTaskKernelFunc 函数。

## 约束说明

无
