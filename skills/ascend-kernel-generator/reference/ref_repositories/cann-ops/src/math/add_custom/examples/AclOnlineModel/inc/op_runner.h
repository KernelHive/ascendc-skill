/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file op_runner.h
 */
#ifndef OP_RUNNER_H
#define OP_RUNNER_H

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "common.h"
#include "operator_desc.h"

/**
 * Op Runner
 */
class OpRunner {
public:
    /**
     * @brief Constructor
     * @param [in] opDesc: op description
     */
    explicit OpRunner(OperatorDesc *opDesc);

    /**
     * @brief Destructor
     */
    virtual ~OpRunner();

    /**
     * @brief Init op runner
     */
    bool Init();

    /**
     * @brief Get number of inputs
     * @return number of inputs
     */
    const size_t NumInputs();

    /**
     * @brief Get number of outputs
     * @return number of outputs
     */
    const size_t NumOutputs();

    /**
     * @brief Get input size by index
     * @param [in] index: input index
     * @return size of the input
     */
    const size_t GetInputSize(size_t index) const;
    const size_t GetInputNumDims(size_t index) const;
    aclDataType GetInputDataType(size_t index) const;
    aclFormat GetInputFormat(size_t index) const;

    /**
     * @brief Get output size by index
     * @param [in] index: output index
     * @return size of the output
     */
    size_t GetOutputSize(size_t index) const;
    const size_t GetOutputNumDims(size_t index) const;
    aclDataType GetOutputDataType(size_t index) const;
    aclFormat GetOutputFormat(size_t index) const;

    /**
     * @brief Get input element count by index
     * @param i[in] ndex: input index
     * @return element count of the input
     */
    size_t GetInputElementCount(size_t index) const;

    /**
     * @brief Get output element count by index
     * @param [in] index: output index
     * @return element count of the output
     */
    size_t GetOutputElementCount(size_t index) const;

    /**
     * @brief Get input shape by index
     * @param [in] index: input index
     * @return shape of the output
     */
    std::vector<int64_t> GetInputShape(size_t index) const;

    /**
     * @brief Get output shape by index
     * @param [in] index: output index
     * @return shape of the output
     */
    std::vector<int64_t> GetOutputShape(size_t index) const;

    /**
     * @brief Get input buffer(host memory) by index
     * @tparam T: data type
     * @param [in] index: input index
     * @return host address of the input
     */
    template <typename T> T *GetInputBuffer(size_t index)
    {
        if (index >= numInputs_) {
            ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
            return nullptr;
        }
        return reinterpret_cast<T *>(hostInputs_[index]);
    }

    /**
     * @brief Get output buffer(host memory) by index
     * @tparam T: data type
     * @param [in] index: output index
     * @return host address of the output
     */
    template <typename T> const T *GetOutputBuffer(size_t index)
    {
        if (index >= numOutputs_) {
            ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
            return nullptr;
        }

        return reinterpret_cast<T *>(hostOutputs_[index]);
    }

    /**
     * @brief Print readable input by index
     * @param [in] index: input index
     * @param [in] elementsPerRow: number of elements per row
     */
    void PrintInput(size_t index, size_t elementsPerRow = 16);

    /**
     * @brief Print readable output by index
     * @param [in] index: output index
     * @param [in] elementsPerRow: number of elements per row
     */
    void PrintOutput(size_t index, size_t elementsPerRow = 16);

    /**
     * @brief Compile static op
     * @return compile result
     */
    bool CompileStaticOp();

    /**
     * @brief Compile dynamic op
     * @return compile result
     */
    bool CompileDynamicOp();

    /**
     * @brief Run op
     * @return run result
     */
    bool RunOp();

private:
    size_t numInputs_;
    size_t numOutputs_;

    std::vector<aclDataBuffer *> inputBuffers_;
    std::vector<aclDataBuffer *> outputBuffers_;

    std::vector<void *> devInputs_;
    std::vector<void *> devOutputs_;

    std::vector<void *> hostInputs_;
    std::vector<void *> hostOutputs_;

    std::vector<aclTensor *> inputTensor_;
    std::vector<aclTensor *> outputTensor_;
    OperatorDesc *opDesc_;
};

#endif // OP_RUNNER_H
