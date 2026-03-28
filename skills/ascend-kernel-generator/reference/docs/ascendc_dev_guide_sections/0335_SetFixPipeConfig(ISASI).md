##### SetFixPipeConfig(ISASI)

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

DataCopy（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中 tensor 量化参数。

## 函数原型

```cpp
template <typename T>
__aicore__ inline void SetFixPipeConfig(const LocalTensor<T> &reluPre, const LocalTensor<T> &quantPre,
bool isUnitFlag = false)

template <typename T, bool setRelu = false>
__aicore__ inline void SetFixPipeConfig(const LocalTensor<T> &preData, bool isUnitFlag = false)
```

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型 |
| setRelu | 针对设置一个 tensor 的情况，当 setRelu 为 true 时，设置 reluPre；反之设置 quantPre。当前仅支持设置为 false |

### 参数说明

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| reluPre | 输入 | 源操作数，relu tensor，类型为 LocalTensor，支持的 TPosition 为 C2PIPE2GM。预留参数，暂未启用，为后续的功能扩展做保留，传入一个空 LocalTensor 即可 |
| quantPre | 输入 | 源操作数，quant tensor，量化操作时参与计算的 tensor，类型为 LocalTensor，支持的 TPosition 为 C2PIPE2GM |
| isUnitFlag | 输入 | UnitFlag 配置项，类型为 bool。预留参数，暂未启用，为后续的功能扩展做保留，保持默认值 false 即可 |
| preData | 输入 | 支持设置一个 Tensor，通过开关控制是 relu Tensor 还是 quant Tensor，支持的 TPosition 为 C2PIPE2GM。当前仅支持传入 quant Tensor |

## 约束说明

无

## 返回值说明

无

## 调用示例

完整示例可参考完整示例。

```cpp
__aicore__ inline void SetFPC(const LocalTensor<int32_t>& reluPreTensor, const LocalTensor<int32_t>& quantPreTensor)
{
    // reluPreTensor 为空 tensor
    AscendC::SetFixPipeConfig<int32_t>(reluPreTensor, quantPreTensor);
    
    // 等效调用:
    // AscendC::SetFixPipeConfig<int32_t>(quantPreTensor);
}
```
