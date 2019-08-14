#if USE_EXP

#include <vector>
#include <CPUBinary.hpp>
#include <CaffeOp_generated.h>
#include "BackendImpl.h"
#include "exp_add.h"
#include "Interpreter.hpp"
#include <errorcode.h>
#include <iostream>
#include <MNN_generated.h>
#include <math.h>
#include <algorithm>
#include "CPUBackend.hpp"
#include "Macro.h"
#include <cstring>

namespace mslite {

#define NUM_ADD_PARAMS (6)

    class AddKernel : public Kernel {
    public:
        AddKernel(const std::string &opName, const std::string &attr, BackendImpl *b);

        virtual ~AddKernel();

        virtual OpFunc GetOpFunc();

        int Run(std::vector<MSTensor *> &);

    protected:
        int ParseParams();

        std::unique_ptr<MNN::Execution> ptr_;
        BackendImpl *backend_ = nullptr;
        flatbuffers::FlatBufferBuilder fbb_;
        std::vector<int> inp_dims_ = {0, 0, 0, 0};
        int32_t layout_;
        std::vector<MNN::Tensor *> inputs_;
        std::vector<MNN::Tensor *> outputs_;



    };

    AddKernel::AddKernel(const std::string &opName, const std::string &attr, BackendImpl *b)
            : Kernel(opName, attr), backend_(b) {
        inputs_= std::vector<MNN::Tensor *> (2);
        outputs_ = std::vector<MNN::Tensor *>(1);
    }

    AddKernel::~AddKernel() {
        backend_ = nullptr;
    }

    OpFunc AddKernel::GetOpFunc() {
        using std::placeholders::_1;
        return std::bind(&AddKernel::Run, this, _1);
    }

    int AddKernel::ParseParams() {
        if (params_.size() != NUM_ADD_PARAMS) {
            return RET_ERROR;
        }
        layout_ = params_[0];
        assert(layout_ == FORMAT_NCHW);

        inp_dims_[0] = params_[2];    // N
        inp_dims_[1] = params_[3];    // C
        inp_dims_[2] = params_[4];    // H
        inp_dims_[3] = params_[5];    // W
        return RET_OK;
    }

    int AddKernel::Run(std::vector<MSTensor *> &tensors) {

        int errCode = RET_OK;
        if (backend_) {
            errCode = ParseParams();
            if (tensors.size() != EXP_PARAMS_NUM)
                return RET_ERROR;

            // onCopyBuffer
            MNN::BinaryOpBuilder bob(fbb_);
            if (type_!="Add")
                return RET_ERROR;

            bob.add_opType(MNN::BinaryOpOperation_ADD);

            auto binary = bob.Finish();
            auto name = fbb_.CreateString(op_);
            auto iv = fbb_.CreateVector(std::vector<int>({0, 1}));
            auto ov = fbb_.CreateVector(std::vector<int>({2}));

            MNN::OpBuilder builder(fbb_);
            builder.add_type(MNN::OpType_BinaryOp);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);
            builder.add_main_type(MNN::OpParameter_BinaryOp);
            builder.add_main(flatbuffers::Offset<void>(binary.o));

            auto opFull = builder.Finish();
            fbb_.Finish(opFull);
            auto op = flatbuffers::GetRoot<MNN::Op>(fbb_.GetBufferPointer());

            inputs_[0] = tensors[EXP_INPUT1_POS]->get();
            inputs_[1] = tensors[EXP_INPUT2_POS]->get();
            outputs_[0] = tensors[EXP_OUTPUT_POS]->get();

            ptr_ = std::unique_ptr<MNN::Execution>(backend_->backend.onCreate(inputs_, outputs_, op));
        }
        if (ptr_ != nullptr) {

            inputs_[0] = tensors[EXP_INPUT1_POS]->get();
            inputs_[1] = tensors[EXP_INPUT2_POS]->get();
            outputs_[0] = tensors[EXP_OUTPUT_POS]->get();
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

    KernelImpl CreateAddKernel(const std::string &opName, const std::string &attr, BackendImpl *b) {
        if (b == nullptr) {
            return nullptr;
        }
        KernelImpl kernel = std::shared_ptr<Kernel>(new AddKernel(opName, attr, b));
        return kernel;
    }

}    // namespace mslite

#endif    // EXP
