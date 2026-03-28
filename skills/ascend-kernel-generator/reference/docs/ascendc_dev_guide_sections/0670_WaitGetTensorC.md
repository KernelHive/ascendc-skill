###### WaitGetTensorC

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

当使用 GetTensorC 异步接口将结果矩阵从 GM 拷贝到 UB，且 UB 后续需要进行 Vector 计算时，需要调用 WaitGetTensorC 进行同步。

## 函数原型

```cpp
__aicore__ inline void WaitGetTensorC()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

当使能 MixDualMaster（双主模式）场景时，即模板参数 enableMixDualMaster 设置为 true，不支持使用该接口。

## 调用示例

```cpp
// 异步模式样例
mm.template Iterate<false>();
// 其他操作
for (int i = 0; i < singleCoreM/baseM * singleCoreN/baseN; ++i) {
    mm.template GetTensorC<false>(ubCmatrix);
    mm.WaitGetTensorC();
    // Vector 操作
}
```
