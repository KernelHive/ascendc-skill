# GCD
## 支持的产品型号

- Atlas A2训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  `GCD`算子返回两个整数张量元素的最大公约数。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">GCD</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="2" align="center">算子输入</td>
     
    <td align="center">x1</td><td align="center">tensor</td><td align="center">int16,int32,int64</td><td align="center">ND</td></tr>  
    <td align="center">x2</td><td align="center">tensor</td><td align="center">int16,int32,int64</td><td align="center">ND</td></tr>  
    
    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">int16,int32,int64</td><td align="center">ND</td></tr>  
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gcd</td></tr>  
  </table>

## 约束与限制

- 输入输出支持5维张量，支持广播操作
- 在广播情况下性能可能受影响

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用GCD算子。</td>
    </tr>
</table>
