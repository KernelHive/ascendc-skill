###### ReduceSum

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

对一个多维向量按照指定的维度进行数据累加。

定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。如下图所示，对shape为(2, 3)的二维矩阵进行运算，指定在第一维计算数据的累加，输出结果为[5, 7, 9]；指定在第二维计算数据的累加，输出结果为[6, 15]。

图 15-65 ReduceSum 按第一个维度计算示例

图 15-66 ReduceSum 按最后一个维度计算示例

## 函数原型

### 通过sharedTmpBuffer入参传入临时空间

```cpp
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceSum(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t srcShape[], bool srcInnerPad)
```

### 接口框架申请临时空间

```cpp
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceSum(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
const uint32_t srcShape[], bool srcInnerPad)
```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者通过sharedTmpBuffer入参传入和接口框架申请两种方式。

- **通过sharedTmpBuffer入参传入**：使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
- **接口框架申请临时空间**：开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过GetReduceSumMaxMinTmpSize中提供的接口获取需要预留空间范围的大小。

## 参数说明

### 表 15-800 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：float。<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：float。 |
| pattern | 用于指定ReduceSum计算轴，包括Reduce轴和Normal轴。pattern由与向量维度数量相同的A、R字母组合形成，字母A表示Normal轴，R表示Reduce轴。例如，AR表示对二维向量进行ReduceSum计算：第一维是Normal轴，第二维是Reduce轴，即对第二维进行数据累加的计算。<br>pattern是定义在AscendC::Pattern::Reduce命名空间下的结构体，其成员变量用户无需关注。<br>pattern当前只支持取值为AR和RA。 |
| isReuseSource | 是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。<br>设置为true，则本接口内部计算时复用src的内存空间，节省内存空间；设置为false，则本接口内部计算时不复用src的内存空间。<br>isReuseSource的使用样例请参考更多样例。 |

### 表 15-801 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dstTensor | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>用于ReduceSum内部复杂计算时存储中间变量，由开发者提供。<br>临时空间大小BufferSize的获取方式请参考GetReduceSumMaxMinTmpSize。 |
| srcShape | 输入 | uint32_t类型的数组，表示源操作数的shape信息。该shape的维度必须和模板参数pattern的维度一致，例如，pattern为AR，该shape维度只能是二维。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，当前只支持二维shape。<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，当前只支持二维shape。 |
| srcInnerPad | 输入 | 表示实际需要计算的最内层轴数据是否32Bytes对齐。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，当前只支持true。<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，当前只支持true。 |

## 返回值说明

无

## 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。
- 不支持源操作数与目的操作数地址重叠。
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
- 内部算法不处理累加计算时的数据溢出，溢出场景不保证接口精度。

## 调用示例

```cpp
AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();
AscendC::LocalTensor<uint8_t> tmp = tbuf.Get<uint8_t>();
uint32_t shape[] = { 2, 8 };
constexpr bool isReuse = true;
AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, isReuse>(dstLocal, srcLocal, tmp, shape, true);
```

结果示例如下：

- 输入输出的数据类型为float
- 输入数据(src):
  ```
  [[ 0.0 4.0 2.0 0.0 -1.0 2.0 -1.0 7.0],
   [ 0.0 1.0 -9.0 2.0 2.0 2.0 8.0 3.0]]
  ```
- 输入pattern：AR
- 输入shape：(2,8)
- 输出数据(dst): [13.0 9.0]
