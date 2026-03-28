## 概述

通过aclnn调用的方式调用BatchMatMulV3算子。

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
aclnnStatus aclnnBatchMatMulGetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor)
// 执行算子
aclnnStatus aclnnBatchMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
```

其中aclnnBatchMatMulGetWorkspaceSize为第一段接口，主要用于计算本次API调用计算过程中需要多少的workspace内存。获取到本次API计算需要的workspace大小之后，按照workspaceSize大小申请Device侧内存，然后调用第二段接口aclnnBatchMatMul执行计算。具体参考[AscendCL单算子调用](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp)>单算子API执行 章节。

## 运行样例算子
  **请确保已根据算子包编译部署步骤完成本算子的编译部署动作。**
  
  - 进入样例代码所在路径
  
  ```bash
  cd ${git_clone_path}/cann-ops/src/matmul/batch_mat_mul_v3/examples/AclNNInvocationNaive
  ```
  
  - 环境变量配置
    
    需要设置环境变量，以arm为例
    
    ```bash
    export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
    export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/lib64
    ```
    以上需要根据CANN包路径修改。
  - 样例执行
    
    样例执行过程中会自动生成测试数据，然后编译与运行aclnn样例，最后打印运行结果。
    
    ```bash
    python3 gen_data.py
    mkdir -p build
    cd build
    cmake .. && make
    ./execute_test_op
    cd ..
    python3 verify_result.py output/output.bin output/golden.bin
    ```
    
    用户亦可参考run.sh脚本进行编译与运行。
    
    ```bash
    bash run.sh
    ```

## 更新说明

| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/04/15 | 新增本readme |




