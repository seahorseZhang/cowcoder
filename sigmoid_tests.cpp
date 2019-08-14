/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: The unit test for sigmoid
 * Author: linbaizhu l00439096
 * Create: 2019-08-01
 */
#if USE_EXP

#include <gtest/gtest.h>
#include <BackendImpl.h>
#include <exp_sigmoid.h>
#include <errorcode.h>
#include "tests_common.h"
#include <math.h>

using namespace mslite;

const int SIGMOID_SEED = 333;
const int PARAM_LENGTH = 1024;

/* the code is taken from openVINO test suite */
/* ToDo: update reference functionality!      */

template <typename T>
T logistic_fwd(T s)
{
    T v = ::expf(s);
    return v / (v + 1);
}

template <typename data_t>
bool ReferenceSigmoid(MSTensor& input,
                      MSTensor& output)
{

    auto dims = input->shape();

    size_t IN = dims[NCHW_DIM_N];
    size_t IC = dims[NCHW_DIM_C];
    size_t IH = dims[NCHW_DIM_H];
    size_t IW = dims[NCHW_DIM_W];

    const data_t *src_data = input->host<data_t>();
    data_t *dst_data = output->host<data_t>();

    for (uint32_t n = 0; n < IN; n++) {
        for (uint32_t c = 0; c < IC; c++) {
            for (uint32_t h = 0; h < IH; h++) {
                for (uint32_t w = 0; w < IW; w++) {
                    uint32_t oidx = n * IC * IH * IW
                                    + c * IH * IW
                                    + h * IW
                                    + w;
                    dst_data[oidx] = logistic_fwd(src_data[oidx]);
                }
            }
        }
    }

    return true;
}


class SigmoidTest : public ::testing::TestWithParam<std::tuple<std::vector<int64_t>, Format, DataType>> {
protected:
    void initTensor(std::vector<int64_t>& dims,
                    Format layout,
                    DataType type,
                    MSTensor& tensor)
    {
        auto res = mslite::MSTensorUtil::MallocDesc(dims, type, layout, tensor);
        ASSERT_EQ(res,  mslite::RET_OK);
        res = MSTensorUtil::MallocData(tensor);
        ASSERT_EQ(res,  mslite::RET_OK);
        auto ptr = mslite::MSTensorUtil::GetData(tensor);
        ASSERT_NE(ptr, nullptr);
        fillRandomData(tensor);
    }
    void SetUp()
    {
        std::srand(SIGMOID_SEED);
        auto param = GetParam();
        dims          = std::get<0>(param);
        auto layout   = std::get<1>(param);
        auto dataType = std::get<2>(param);

        char buffer[PARAM_LENGTH] = {0};

        EXPECT_EQ(dims.size(), 4ul);

        sprintf(buffer, "Activation_%d_4_2_%d_%d_%d_%d",
            layout, (int32_t)dims[0], (int32_t)dims[1], (int32_t)dims[2], (int32_t)dims[3]);

        attrs_ = buffer;

        /* input/output initialization */
        initTensor(dims, layout, dataType, refInput);
        initTensor(dims, layout, dataType, refOutput);

        /* backend dependent kernels implementation */
        input   = initInternalTensor(refInput, backend);
        output  = initInternalTensor(refOutput, backend);
    }
    void TearDown()
    {
        auto res = MSTensorUtil::FreeData(input);
        ASSERT_EQ(res,  mslite::RET_OK);
        res = MSTensorUtil::FreeData(output);
        ASSERT_EQ(res,  mslite::RET_OK);
        res = MSTensorUtil::FreeData(refOutput);
        ASSERT_EQ(res,  mslite::RET_OK);
        res = MSTensorUtil::FreeData(refInput);
        ASSERT_EQ(res,  mslite::RET_OK);
    }
    MSTensor input;
    MSTensor output;

    std::vector<int64_t> dims;
    MSTensor refInput;
    MSTensor refOutput;

    BackendImpl backend;
    std::string attrs_;
};

TEST_P(SigmoidTest, RunSigmoid)
{
    auto kernel = CreateSigmoidKernel("test_kernel", attrs_, &backend);
    EXPECT_NE(kernel, nullptr);
    auto callback = kernel->GetOpFunc();
    EXPECT_NE(callback, nullptr);
    std::vector<MSTensor *> params(EXP_SIGMOID_PARAMS_NUM);
    EXPECT_EQ(static_cast<int>(params.size()), EXP_SIGMOID_PARAMS_NUM);

    params[EXP_SIGMOID_INPUT_POS]   = &input;
    params[EXP_SIGMOID_OUTPUT_POS]  = &output;

    auto res = callback(params);
    EXPECT_EQ(res, 0);

    auto refRes = ReferenceSigmoid<float>(refInput, refOutput);
    EXPECT_TRUE(refRes);

    MSTensor outTensorNchw(new MNN::Tensor(output.get(), MNN::Tensor::CAFFE, true));
    backend.backend.onCopyBuffer(output.get(), outTensorNchw.get());

    const float epsilon = 1e-3f;
    auto executeRes = compareTensors(refOutput, outTensorNchw, epsilon);
    EXPECT_TRUE(executeRes);
}


static const std::vector<Format> g_availableLayouts = {
    FORMAT_NCHW,
};

static const std::vector<DataType> g_availableTypes = {
    DT_FLOAT,
};

// common test
INSTANTIATE_TEST_CASE_P(SigmoidTests_common, SigmoidTest,
                        testing::Combine(testing::Values(std::vector<int64_t>{1, 1, 64, 64},
                                                         std::vector<int64_t>{1, 3, 32, 32},
                                                         std::vector<int64_t>{1, 3, 128, 256},
                                                         std::vector<int64_t>{2, 3, 32, 32},
                                                         std::vector<int64_t>{1, 7, 200, 200},
                                                         std::vector<int64_t>{1, 1, 1917, 2}
                                                         ),
                                         testing::ValuesIn(g_availableLayouts),
                                         testing::ValuesIn(g_availableTypes)));
#endif // USE_EXP
