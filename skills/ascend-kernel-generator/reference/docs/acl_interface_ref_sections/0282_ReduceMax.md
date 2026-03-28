### ReduceMax

功能
计算输入Tensor沿指定维度的最大值。如果keepdims等于1，得到的Tensor的维数与
输入的相同。如果keepdims等于0，那么生成的Tensor会去除被缩减的维度。

输入
data：输入Tensor，数据类型：float16、float、int8、int32、int64、double。

输出
reduced data：输出Tensor，和输入data的数据类型一致。

属性
● axes：数据类型为Int列表；指定计算轴；取值范围：[-r, r-1]，r是输入数据data
的维数。


● keepdims(default is "1")：数据类型为int；是否保留缩减的维度；默认为1（保
留）。

限制与约束
当输入为float16、float、int32类型时，axes属性可以为空；当输入为其他类型时，需
要配置axes属性值。
axes为空时，当前输出不做全维度规约。此时建议修改onnx算子的axes为所有轴；或
者在torch导出onnx图前，在torch模型中使用amax来对所有轴规约（例如，
x.amax(dim=[0, 1, 2])）。

支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
