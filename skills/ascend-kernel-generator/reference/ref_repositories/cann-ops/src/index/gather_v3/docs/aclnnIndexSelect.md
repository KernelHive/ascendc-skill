# aclnnIndexSelect

## 支持的产品型号
- Atlas 推理系列产品
- Atlas 训练系列产品
- Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas 800I A2 推理产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnIndexSelectGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIndexSelect”接口执行计算。

- `aclnnStatus aclnnIndexSelectGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index,  aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnIndexSelect(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## 功能描述

**算子功能：**
从输入Tensor的指定维度dim，按index中的下标序号提取元素，保存到out Tensor中。
例如，对于输入张量 $self=\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}$ 和索引张量 index=[1, 0]，
self.index_select(0, index)的结果： $y=\begin{bmatrix}4 & 5 & 6 \\ 1 & 2 & 3\end{bmatrix}$;

x.index_select(1, index)的结果： $y=\begin{bmatrix}2 & 1\\ 5 & 4\\8 & 7\end{bmatrix}$;

**具体计算过程如下：**
以三维张量为例, shape为(3,2,2)的张量 **self** =$\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$   **index**=[1, 0], self张量dim=0、1、2对应的下标分别是$l、m、n$,  index是一维

dim为0, index_select(0, index)：   I=index[i];  &nbsp;&nbsp;   out$[i][m][n]$ = self$[I][m][n]$

dim为1, index_select(1, index)：   J=index[j];  &nbsp;&nbsp;&nbsp;    out$[l][j][n]$ = self$[l][J][n]$

dim为2, index_select(2, index)：   K=index[k]; &nbsp;  out$[l][m][k]$ = self$[l][m][K]$

## aclnnIndexSelectGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：Device侧的aclTensor，支持非连续的Tensor，数据格式支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW。维度不大于8。
     * Atlas A2 训练系列产品/Atlas A3 训练系列产品：FLOAT、FLOAT16、BFLOAT16、INT64、INT32、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128
     * Atlas 推理系列产品/Atlas 训练系列产品：FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128
  - dim(int64_t, 计算输入)：指定的维度，int64类型，范围[-self.dim(), self.dim() - 1]。
  - index(aclTensor*, 计算输入)：索引，Device侧的aclTensor，数据类型支持INT64、INT32。支持非连续的Tensor，数据格式支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW，且只能是0D或1D（零维的情况：当成是size为1的一维）。index中的索引数据不支持越界。
  - out(aclTensor*, 计算输出)：输出Tensor，Device侧的aclTensor，数据类型同self。维数与self一致。**除dim维长度等于index长度外，其他维长度与self相应维一致**。数据格式支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW。
     * Atlas A2 训练系列产品/Atlas A3 训练系列产品：FLOAT、FLOAT16、BFLOAT16、INT64、INT32、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128
     * Atlas 推理系列产品/Atlas 训练系列产品：FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128
  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 参数self、index、out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. 参数self、index的数据类型不在支持的范围内。
                                        2. dim >= self.dim() 或者 dim < -self.dim()。
                                        3. index维度大于1，报错。
                                        4. self维度大于8，报错。
  ```

## aclnnIndexSelect

- **参数说明：**

  - workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnIndexSelectGetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

无。

## 调用示例

详见[GatherV3自定义算子样例说明算子调用章节](../README.md#算子调用)