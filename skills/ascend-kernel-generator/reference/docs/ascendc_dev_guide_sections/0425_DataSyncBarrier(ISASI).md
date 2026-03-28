###### DataSyncBarrier(ISASI)

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

用于阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束。

## 函数原型

```cpp
template <MemDsbT arg0>
__aicore__ inline void DataSyncBarrier()
```

## 参数说明

**表 15-392 模板参数说明**

| 参数名 | 描述 |
|--------|------|
| arg0 | 模板参数，表示需要等待的内存位置，类型为MemDsbT，可取值：<br>• ALL，等待所有内存访问指令<br>• DDR，等待GM访问指令<br>• UB，等待UB访问指令<br>• SEQ，预留参数，暂未启用，为后续的功能扩展做保留 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
AscendC::Mmad(...);
AscendC::DataSyncBarrier<MemDsbT::ALL>();
AscendC::Fixpipe(...);
```
