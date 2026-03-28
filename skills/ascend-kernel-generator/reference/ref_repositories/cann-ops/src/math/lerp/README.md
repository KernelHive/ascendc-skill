# Lerp自定义算子样例说明
本样例通过Ascend C编程语言实现了Lerp算子。


## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品


## 算子描述
- 功能描述

  `Lerp`算子用对对输入数据（起始值、结束值和插值比例）进行线性插值计算，将结果返回到输出张量。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Lerp</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="4" align="center">算子输入</td>
     
    <tr><td align="center">start</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
    <tr><td align="center">end</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
    <tr><td align="center">weight</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
    
    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
    
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">lerp</td></tr>  
  </table>

## 约束与限制
- start，end，weight，y，out的数据类型只支持float32,float16,bfloat16，数据格式只支持ND

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Lerp算子。</td>
    </tr>
</table>
