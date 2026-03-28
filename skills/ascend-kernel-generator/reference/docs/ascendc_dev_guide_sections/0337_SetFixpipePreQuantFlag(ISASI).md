##### SetFixpipePreQuantFlag(ISASI)

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

DataCopy（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中标量量化参数。

## 函数原型

```cpp
__aicore__ inline void SetFixpipePreQuantFlag(uint64_t config)
```

## 参数说明

**表 15-297 参数说明**

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| config | 输入 | 量化过程中使用到的标量量化参数，类型为uint64_t |

## 返回值说明

无

## 约束说明

无

## 调用示例

完整示例可参考完整示例。

```cpp
uint64_t deqScalar = 11;
AscendC::SetFixpipePreQuantFlag(deqScalar);
```
