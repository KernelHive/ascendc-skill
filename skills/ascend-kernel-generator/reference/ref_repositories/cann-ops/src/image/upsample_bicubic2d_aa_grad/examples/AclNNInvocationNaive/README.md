## 概述

通过aclnn调用的方式调用UpsampleBicubic2dAAGrad算子。

## 目录结构介绍

```
├── AclNNInvocationNaive
│   ├── CMakeLists.txt      // 编译规则文件
│   ├── gen_data.py         // 算子期望数据生成脚本
│   ├── main.cpp            // 单算子调用应用的入口
│   ├── run.sh              // 编译运行算子的脚本
│   └── verify_result.py    // 计算结果精度比对脚本
```

## 代码实现介绍

完成自定义算子的开发部署后，可以通过单算子调用的方式来验证单算子的功能。main.cpp代码为单算子API执行方式。单算子API执行是基于C语言的API执行算子，无需提供单算子描述文件进行离线模型的转换，直接调用单算子API接口。

自定义算子编译部署后，会自动生成单算子API，可以直接在应用程序中调用。算子API的形式一般定义为“两段式接口”，形如：

```cpp
// 获取算子使用的workspace空间大小
aclnnStatus aclnnUpsampleBicubic2dAAGradGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *outputSize, const aclIntArray *inputSize, bool alignCorners, double scalesH, double scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
// 执行算子
aclnnStatus aclnnUpsampleBicubic2dAAGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

其中aclnnUpsampleBicubic2dAAGradGetWorkspaceSize为第一段接口，主要用于计算本次API调用计算过程中需要多少的workspace内存。获取到本次API计算需要的workspace大小之后，按照workspaceSize大小申请Device侧内存，然后调用第二段接口aclnnUpsampleBicubic2dAAGrad执行计算。具体参考[AscendCL单算子调用](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp)>单算子API执行 章节。

## 运行样例算子
**请确保已根据算子包编译部署步骤完成本算子的编译部署动作。**
  
- 进入样例代码所在路径
  
  ```bash
  cd ${git_clone_path}/cann-ops/src/image/upsample_bicubic2d_aa_grad/examples/AclNNInvocationNaive
  ```
  
- 样例执行
    
  样例执行过程中会自动生成测试数据，然后编译与运行aclnn样例，最后打印运行结果。

  ```bash
  bash run.sh
  ```

## 更新说明

| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/01/06 | 新增本readme |
