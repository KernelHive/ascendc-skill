声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Cast

## 支持的产品型号

Atlas A2训练系列产品 / Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：将tensor从源数据类型转换为指定的目标数据类型。

## 实现原理

调用`Ascend C`的`API`接口`Cast`进行实现。

- 对于`Cast`API直接支持的数据类型，直接调用`Cast`API即可。
- 对于部分无法直接使用`Cast`API直接转换的数据类型，如float16->bfloat16，需要利用中间数据进行转换，如float16先转换为float32，再将float32转换为bfloat16。
- 对于部分数据转换后会出现数据上溢的情况，如float16->int8，需要对原始float16数据进行处理（float16->int8的计算流程为：float16先转换为int16，int16对256取模，将结果+128，再对256取模，再-128，最后将其转为int8），其他数据类型的处理方式类似。
- 对于`int8`，`uint8`，`bool`，这三种数据之间的转换，直接利用TQueBind搬运即可，无需进行计算处理。
- 对于其他数据类型转换为`bool`数据类型的转换，如float16->bool，需要先利用`Abs`API取绝对值，然后利用`Mins`与1比较，最后利用`Cast`向上取整为int8，其他数据类型的处理方式类似。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnCastGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCast”接口执行计算。

* `aclnnStatus aclnnCastGetWorkspaceSize(const aclTensor *x, int64_t dstType, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnCast(void *workspace, uint64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnCastGetWorkspaceSize

- **参数说明：**
  
  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、INT32、INT8、UINT8、BOOL、INT64、BFLOAT16、INT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - dstType（int64_t, 算子属性）：必选参数，目标数据类型，支持数据类型为INT64。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、INT32、INT8、UINT8、BOOL、INT64、BFLOAT16、INT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、dstType、out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnCast

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnCastGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- dstType需要与out的数据类型一致，具体可以参考(https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/basicdataapi/atlasopapi_07_00514.html)
- x，out的数据格式只支持ND
- x，out具体支持的数据类型如下表所示

    <table>
    <tr>
    <td></td><td></td><td colspan="9" align="center">目标数据类型(out)</td>
    </tr>
    <tr>
    <td></td><td></td><td align="center">float16</td><td align="center">float32</td><td align="center">int32</td><td align="center">int8</td><td align="center">uint8</td><td align="center">bool</td><td align="center">int64</td><td align="center">bfloat16</td><td align="center">int16</td>
    </tr>
    <tr>
    <td rowspan="9" align="center">源数据类型(x)</td><td align="center">float16</td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">float32</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">int32</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">int8</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">uint8</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">bool</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td><td align="center"></td>
    </tr>
    <tr>
    <td align="center">int64</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center">√</td>
    </tr>
    <tr>
    <td align="center">bfloat16</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center"></td><td align="center"></td>
    </tr>
    <tr>
    <td align="center">int16</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center">√</td><td align="center"></td><td align="center">√</td><td align="center"></td><td align="center"></td>
    </tr>
    </table>


## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Cast</th></tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">type</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">tensor</td><td align="center">float16, float32, int32, int8, uint8, bool, int64, bfloat16, int16</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">tensor</td><td align="center">float16, float32, int32, int8, uint8, bool, int64, bfloat16, int16</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">算子属性</td><td align="center">dstType</td><td align="center">attr</td><td align="center">int64</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cast</td></td></tr>
</table>

## 调用示例

详见[Cast自定义算子样例说明算子调用章节](../README.md#算子调用)
