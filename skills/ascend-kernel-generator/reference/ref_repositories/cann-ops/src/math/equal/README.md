# Equal
## 贡献说明
| 贡献者   | 贡献方              | 贡献算子  | 贡献时间      | 贡献内容      |
|-------|------------------|-------|-----------|-----------|
| 买个橘子去 | 西北工业大学-智能感知交互实验室 | Equal | 2025/1/14 | 新增Equal算子 |

## 支持的产品型号
- Atlas A2训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  `Equal`算子将输入的两个向量数据进行各对应位置的判等运算，返回结果向量。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Equal</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="2" align="center">算子输入</td>
    <tr><td align="center">x1</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,uint8,int8,uint32,int32</td><td align="center">ND</td></tr>  
    <tr><td rowspan="2" align="center">算子输入</td>
    <tr><td align="center">x2</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,uint8,int8,uint32,int32</td><td align="center">ND</td></tr>  
    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">bool</td><td align="center">ND</td></tr>  
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">equal</td></tr>  
  </table>

## 约束与限制
- x,y,out的数据类型仅支持float32,float16,bfloat16,uint8,int8,uint32,int32，数据格式仅支持ND

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Equal算子。</td>
    </tr>
</table>
