/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: The definition of sigmoid
 * Author: linbaizhu l00439096
 * Create: 2019-08-01
 */
#if USE_EXP

#include <vector>
#include <CPUSigmoid.hpp>
#include <CaffeOp_generated.h>
#include "BackendImpl.h"
#include "exp_sigmoid.h"
#include "Interpreter.hpp"
#include "MNN_generated.h"

#include <iostream>
#include <string>
#include "errorcode.h"

namespace mslite {

    const int NUM_SIGMOID_PARAMS = 7;
    const int SIGMOID_FUNCTION_TYPE = 2;
    static const std::string sigmoidType = "Activation";

    class SigmoidKernel : public Kernel {
    public:
        SigmoidKernel(const std::string &opName, const std::string &attr, BackendImpl *b);
        virtual ~SigmoidKernel();
        virtual OpFunc GetOpFunc();
        int Run(std::vector<MSTensor *> &);
    protected:
        int ParseParams();
        std::unique_ptr<MNN::Execution> ptr_;
        BackendImpl* backend_ = nullptr;
        bool init_ = false;
        flatbuffers::FlatBufferBuilder fbb_;
        int func_type_;
        size_t ndim_;
        int32_t layout_ = FORMAT_NCHW;
        std::vector<int> inp_dims_ = {0, 0, 0, 0};
        std::vector<MNN::Tensor*> inputs_;
        std::vector<MNN::Tensor*> outputs_;
    };

    int SigmoidKernel::ParseParams()
    {
        if (type_ != sigmoidType)
            return RET_ERROR;

        func_type_ = params_[2];
        if (func_type_ != SIGMOID_FUNCTION_TYPE)
            return RET_ERROR;

        if (params_.empty())
            return RET_ERROR;

        if (params_.size() != NUM_SIGMOID_PARAMS)
            return RET_ERROR;


        layout_ = params_[0];
        assert(layout_ == FORMAT_NCHW);
        ndim_ = params_[1];
        assert(inp_dims_.size() == ndim_);
        inp_dims_[0] = params_[3];
        inp_dims_[1] = params_[4];
        inp_dims_[2] = params_[5];
        inp_dims_[3] = params_[6];

        return RET_OK;
    }
    SigmoidKernel::SigmoidKernel(const std::string &opName, const std::string &attr, BackendImpl *b)
        : Kernel(opName, attr),
          backend_(b)
    {
        inputs_.resize(1);
        outputs_.resize(1);
    }

    SigmoidKernel::~SigmoidKernel()
    {
        backend_ = nullptr;
        init_    = false;
    }

    OpFunc SigmoidKernel::GetOpFunc()
    {
        using std::placeholders::_1;
        return std::bind(&SigmoidKernel::Run, this, _1);
    }

    int SigmoidKernel::Run(std::vector<MSTensor *> &tensors)
    {
        int errCode = RET_OK;
        if (init_ == false && backend_) {
            if (ParseParams() != RET_OK)
                return RET_ERROR;

            if (tensors.size() != EXP_SIGMOID_PARAMS_NUM)
                return RET_ERROR;

            auto name = fbb_.CreateString("sigmoid");
            auto iv   = fbb_.CreateVector(std::vector<int>({0}));
            auto ov   = fbb_.CreateVector(std::vector<int>({1}));
            MNN::OpBuilder builder(fbb_);
            builder.add_type(MNN::OpType_Sigmoid);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);

            auto opFull = builder.Finish();
            fbb_.Finish(opFull);
            auto op = flatbuffers::GetRoot<MNN::Op>(fbb_.GetBufferPointer());
            inputs_[0] = tensors[EXP_SIGMOID_INPUT_POS]->get();
            outputs_[0] = tensors[EXP_SIGMOID_OUTPUT_POS]->get();
            ptr_ = std::unique_ptr<MNN::Execution>(backend_->backend.onCreate(inputs_, outputs_, op));
        }


        if (ptr_ != nullptr) {
            inputs_[0] = tensors[EXP_SIGMOID_INPUT_POS]->get();
            outputs_[0] = tensors[EXP_SIGMOID_OUTPUT_POS]->get();
            auto res = ptr_->onResize(inputs_, outputs_);
            if (res != MNN::NO_ERROR)
                return RET_ERROR;

            res = ptr_->onExecute(inputs_, outputs_);
            if (res != MNN::NO_ERROR)
                return RET_ERROR;

            res = ptr_->onReleaseCache();
            if (res != MNN::NO_ERROR)
                return RET_ERROR;
        }
        return errCode;
    }

    KernelImpl CreateSigmoidKernel(const std::string &opName, const std::string &attr, BackendImpl *b)
    {
        if (b == nullptr) {
            return nullptr;
        }
        KernelImpl kernel = std::shared_ptr<Kernel>(new SigmoidKernel(opName, attr, b));
        return kernel;
    }

}

#endif // EXP
