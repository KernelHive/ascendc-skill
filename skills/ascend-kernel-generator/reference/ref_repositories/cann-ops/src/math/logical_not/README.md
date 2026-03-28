# LogicalNot

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|----|----|----|------|------|
| 周世星 | 浙江工业大学-智能计算研究所 | LogicalNot | 2025/06/24 | 新增LogicalNot算子，实现了逻辑否功能。 |

## 支持的产品型号

- Atlas A2训练系列产品
   
产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 算子描述

- 功能描述    

  `LogicalNot` 算子对输入的布尔值进行逻辑非运算（取反操作）。

- 计算公式：

  $$
  y = \neg x
  $$

- 原型信息    

   <table>
      <tr>
         <th align="center">算子类型(OpType)</th>
         <th colspan="4" align="center">LogicalNot</th>
      </tr>
      <tr>
         <td align="center"></td>
         <td align="center">name</td>
         <td align="center">Type</td>
         <td align="center">data type</td>
         <td align="center">format</td>
      </tr>
      <tr>
         <td rowspan="2" align="center">算子输入</td>
      </tr>
      <tr>
         <td align="center">x</td>
         <td align="center">tensor</td>
         <td align="center">bool</td>
         <td align="center">ND</td>
      </tr>
      <tr>
         <td rowspan="1" align="center">算子输出</td>
         <td align="center">y</td>
         <td align="center">tensor</td>
         <td align="center">bool</td>
         <td align="center">ND</td>
      </tr>
      <tr>
         <td rowspan="1" align="center">核函数名</td>
         <td colspan="4" align="center">logical_not</td>
      </tr>
   </table>


## 约束与限制

- 无

## 算子使用

编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

### 编译部署

  - 进入到仓库目录
    ```bash
    cd ${git_clone_path}/cann-ops
    ```

  - 执行编译
    ```bash
    bash build.sh -n logical_not
    ```

  - 部署算子包
    ```bash
    bash build_out/CANN-custom_ops-<cann_version>-linux.<arch>.run
    ```

### 运行验证

跳转到对应调用方式目录，参考Readme进行算子运行验证。

<table>
    <th>调用方式</th>
    <th>链接</th>
    <tr>
        <td>aclnn单算子调用</td>
        <td>
            <a href="./examples/AclNNInvocationNaive" >AclNNInvocationNaive</a>
        </td>
    </tr>
</table>
