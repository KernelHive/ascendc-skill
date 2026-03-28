# AddCustom 

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|----|----|----|------|------|
| 章鹏飞 | CANN生态 | AddCustom | 2024/11/29 | 新增AddCustom算子。|
| 陈辉| CANN生态 | AddCustom | 2025/6/20 | 修改AddCustom算子支持非对齐shape。|

## 支持的产品型号

- Atlas 训练系列产品
- Atlas 推理系列产品
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 算子描述

- 功能描述    

  AddCustom算子提供加法的计算功能。

- 原型信息    

    <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">AddCustom</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="3" align="center">算子输入</td>

    <tr><td align="center">x1</td><td align="center">tensor</td><td align="center">float32,float16,int8</td><td align="center">ND</td></tr> 

    <tr><td align="center">x2</td><td align="center">tensor</td><td align="center">float32,float16,int8</td><td align="center">ND</td></tr> 

    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,int8</td><td align="center">ND</td></tr>  
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>  
    </table>

## 约束与限制

- x，y，out的数据类型只支持FLOAT16,FLOAT32,INT8，数据格式只支持ND

## 算子使用

使用该算子前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

### 编译部署

  - 进入到仓库目录
    ```bash
    cd ${git_clone_path}/cann-ops
    ```

  - 执行编译
    ```bash
    bash build.sh
    ```

  - 部署算子包
    ```bash
    bash build_out/CANN-custom_ops-<cann_version>-linux.<arch>.run
    ```

### 运行验证

跳转到对应调用方式目录，参考Readme进行算子运行验证。

<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用AddCustom算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/AclOfflineModel"> AclOfflineModel</td><td>通过aclopExecuteV2调用的方式调用AddCustom算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/AclOnlineModel"> AclOnlineModel</td><td>通过aclopCompile调用的方式调用AddCustom算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/CppExtensions"> CppExtensions</td><td>Pybind方式调用AddCustom算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/PytorchInvocation"> PytorchInvocation</td><td>通过pytorch调用的方式调用AddCustom算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/TensorflowInvocation"> TensorflowInvocation</td><td>通过tensorflow调用的方式调用AddCustom算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/ATBInvocation">ATBInvocation</td><td>通过ATB调用的方式调用AddCustom算子。</td>
    </tr>

</table>