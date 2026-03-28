## 如何使用 SPM Buffer

## 功能介绍

SPM Buffer功能支持对Unified Buffer的暂存操作。当Unified Buffer内存有溢出风险时，可以将Unified Buffer的数据先拷贝到SPM(Spill Memory) Buffer暂存，具体使用时再取回。

## 使用方法

使用方法如下：

1. 调用 `InitSpmBuffer` 接口完成初始化操作
2. 调用 `WriteSpmBuffer` 接口完成数据的拷贝暂存操作
3. 需要使用暂存的数据时，调用 `ReadSpmBuffer` 接口完成数据的取回操作

**存储位置支持：**

- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持暂存到workspace
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，仅支持暂存到workspace
- 针对Atlas 训练系列产品、Atlas 推理系列产品，可以暂存到workspace或者L1 Buffer

## 使用样例

### 1. 初始化操作

**暂存到workspace初始化：**

```cpp
AscendC::TPipe pipe;
int len = 1024;
AscendC::GlobalTensor<half> workspace_gm;
auto usrWorkspace = AscendC::GetUserWorkspace(workspace);
// 此处的usrWorkspace为用户自定义的workspace,类型为half,有len个元素
workspace_gm.SetGlobalBuffer((__gm__ half *)usrWorkspace, len);
auto gm = workspace_gm[AscendC::GetBlockIdx() * len];
pipe.InitSpmBuffer(gm, len * sizeof(half));
```

**暂存到L1 Buffer初始化：**

```cpp
TPipe pipe;
int len = 1024; // 设置spm buffer为1024个类型为T的数据
pipe.InitSpmBuffer(len * sizeof(T));
```

### 2. 暂存和取回操作

```cpp
TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
int dataSize = 32; // 假设T为half类型，从ub上申请一块内存32 * sizeof(half)字节
int offset = 32; // 拷贝到spmBuffer时偏移32字节
pipe.InitBuffer(inQueueSrcVecIn, 1, dataSize);
LocalTensor<half> writeLocal = inQueueSrcVecIn.AllocTensor<half>();
DataCopyParams copyParams{1, 2, 0, 0}; // 从ub上搬运一块长度为2个datablock的数据，一个datablock 32byte
pipe.WriteSpmBuffer(writeLocal, copyParams, offset);
pipe.ReadSpmBuffer(writeLocal, copyParams, offset);
```

## 使用约束

无
