###### SetSingleShape

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

设置Matmul单核计算的形状singleCoreM、singleCoreN、singleCoreK，单位为元素。用于运行时修改shape，比如复用Matmul对象来处理尾块。与SetTail接口功能一致，建议使用本接口。

## 函数原型

```cpp
__aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK)
```

## 参数说明

**表 15-629 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| singleM | 输入 | 设置的singleCoreM大小，单位为元素 |
| singleN | 输入 | 设置的singleCoreN大小，单位为元素 |
| singleK | 输入 | 设置的singleCoreK大小，单位为元素 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.SetSingleShape(tailM,tailN,tailK); // 如果是尾核，需要调整singleCoreM/singleCoreN/singleCoreK
mm.IterateAll(gm_c);
```
