###### ClearBias

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

DisableBias接口与该接口的功能相同，建议使用DisableBias。

清除Bias标志位，表示Matmul计算时没有Bias参与。如果在调用Init时配置了TCubeTiling结构中的isBias参数来使能Bias，调用该接口后，会清除Bias标志位，不再使能Bias。

## 函数原型

```cpp
__aicore__ inline void ClearBias()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.ClearBias(); // 清除tiling中的Bias标志位
mm.IterateAll(gm_c);
```
