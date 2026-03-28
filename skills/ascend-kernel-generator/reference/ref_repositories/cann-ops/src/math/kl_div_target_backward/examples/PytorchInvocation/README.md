## 概述
本样例展示了自定义算子分别向pytorch注册eager模式与torch.compile模式的注册样例，eager模式与torch.compile模式的介绍参考：[Link](https://pytorch.org/get-started/pytorch-2.0)。

## 目录结构介绍
```
├── PytorchInvocation                         //通过pytorch调用的方式调用KlDivTargetBackward算子
│   ├── op_plugin_patch                       // op plugin实现目录
│   │   ├── KlDivTargetBackwardKernelNpuApi.cpp         // pytorch eager模式下算子实现，其中调用aclnn接口
│   │   └── op_plugin_functions.yaml          // pytorch注册的算子原型描述声明文件
│   ├── run_op_plugin.sh                      // 编译pytorch_npu并执行用例的脚本
│   ├── test_ops_custom.py                    // 执行eager模式下算子用例脚本
│   └── test_ops_custom_register_in_graph.py  // 执行torch.compile模式下用例脚本
```

## 样例脚本run_op_plugin.sh关键步骤解析

  - 下载PTA源码仓，并更新submodule,其中torchair、op_plugin是本示例中依赖的submodule
    ```bash
    git clone https://gitee.com/ascend/pytorch.git
    git submodule init
    git submodule update
    git checkout -b v2.1.0 origin/v2.1.0
    ```

  - PTA自定义算子注册, 本样例通过向op_plugin_functions.yaml中写入算子原型的方式注册pytorch自定义算子
    ```bash
    FUNCTION_REGISTE_FIELD="op_plugin_patch/op_plugin_functions.yaml"
    FUNCTION_REGISTE_FILE="${PYTORCH_DIR}/third_party/op-plugin/op_plugin/config/v2r1/op_plugin_functions.yaml"
    line="  - func: npu_kl_div_target_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, bool log_target) -> Tensor"
    if ! grep -q "\  $line" $FUNCTION_REGISTE_FILE; then
        sed -i "/custom:/r   $FUNCTION_REGISTE_FIELD" $FUNCTION_REGISTE_FILE
    fi
    ```

  - 编译PTA插件并安装，其中需要将编写好的单算子kernel文件KlDivTargetBackwardKernelNpuApi.cpp，拷贝至op-plugin目录下编译。
    (注：单算子kernel文件中调用aclnnKlDivTargetBackward接口，需要提前完成Ascend C的算子注册)
    ```bash
    cp -rf op_plugin_patch/*.cpp ${PYTORCH_DIR}/third_party/op-plugin/op_plugin/ops/v2r1/opapi
    export DISABLE_INSTALL_TORCHAIR=FALSE
    cd ${PYTORCH_DIR}
    (bash ci/build.sh --python=${PYTHON_VERSION} --disable_rpc ; pip uninstall torch-npu -y ; pip3 install dist/*.whl)
    ```

## 运行样例算子
该样例脚本基于Pytorch2.1、python3.9 运行
### 1.编译部署自定义算子
参考[编译算子工程](../../README.md#编译部署自定义算子)。

### 2.pytorch调用的方式调用样例运行

  - 进入到样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/src/math/kl_div_target_backward/examples/PytorchInvocation
    ```

  - 样例执行

    样例执行过程中会自动生成测试数据，然后运行pytorch样例，最后检验运行结果。具体过程可参见run_op_plugin.sh脚本。
    ```bash
    bash run_op_plugin.sh
    ```

#### 其他样例运行说明
  - 环境安装完成后，样例支持单独执行：eager模式与compile模式的测试用例
    - 执行pytorch eager模式的自定义算子测试文件
      ```bash
      python3 test_ops_custom.py
      ```

## 其他说明
更加详细的Pytorch适配算子开发指导可以参考[Link](https://gitee.com/ascend/op-plugin/wikis)中的“算子适配开发指南”。

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/04/28 | 新增本readme |