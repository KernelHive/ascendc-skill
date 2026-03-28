## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  对输入张量中的每个元素`x`计算`log`函数，并将结果返回到输出张量。

- 原型信息

<table border="1">
  <tr>
    <th align="center" rowspan="2">算子类型(OpType)</th>
    <th colspan="4" align="center">Log</th>
  </tr>
  <tr>
    <th align="center">name</th>
    <th align="center">Type</th>
    <th align="center">data type</th>
    <th align="center">format</th>
  </tr>
  
  <tr>
    <td rowspan="1" align="center">算子输入</td>
    <td align="center">x</td>
    <td align="center">tensor</td>
    <td align="center">bfloat16,float32,float16</td>
    <td align="center">ND</td>
  </tr>
  
  <tr>
    <td rowspan="3" align="center">算子属性</td>
    <td align="center">base</td>
    <td align="center">float</td>
    <td align="center">float</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">scale</td>
    <td align="center">float</td>
    <td align="center">float</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">shift</td>
    <td align="center">float</td>
    <td align="center">float</td>
    <td align="center">ND</td>
  </tr>
  
  <tr>
    <td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td>
    <td align="center">tensor</td>
    <td align="center">bfloat16,float32,float16</td>
    <td align="center">ND</td>
  </tr>
  
  <tr>
    <td align="center">核函数名</td>
    <td colspan="4" align="center">log</td>
  </tr>
</table>

## 约束与限制
- x，y的数据类型只支持bfloat16,float32,float16，数据格式只支持ND
- base,scale,shift的数据类型只支持float

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Log算子。</td>
    </tr>
</table>
