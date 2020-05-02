/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model.h"
#include "loss.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>

namespace fasttext {

    void Model::update(
            const int32_t & input,
            const std::vector<int32_t>& targets,
            int32_t targetIndex,
            int32_t secTargetIdex,
            real lr,
            State& state) {
        computeHidden(input, state);
        Vector& grad = state.inputGrad;
        grad.zero();
        loss_->forward(targets, targetIndex, secTargetIdex, state, lr, true);
        state.incrementNExamples();
        /*riemannian gradient update*/
        real projectScale = state.inputVec.dotMul(state.inputGrad,1.0);
        Vector Z(wo_->size(1));
        Z.zero();Z.addVector(state.inputGrad,1.0);
        Z.addVector(state.inputVec,-1.0*projectScale);
        Z.mul(-1.0 * lr * (1.0+projectScale/state.inputGrad.norm()));
        wi_->addVectorToRow(Z,input,1.0);
        wi_->scalerMulRow(1.0 / wi_->l2NormRow(input), input);
    }

    void Model::computeHidden(const int32_t & input, State& state)
    const {
        Vector& hidden = state.inputVec;
        hidden.zero();
        hidden.addRow(*wi_, input);
        hidden.mul(1.0/hidden.norm());
    }

    void Model::State::incrementNExamples() {
        nexamples_++;
    }

    void Model::State::incrementLoss(real& tmpLossFirst, real& tmpLossSecond) {
        lossFirst_ += tmpLossFirst;
        lossSecond_ += tmpLossSecond;
    }

    real Model::State::getFirstLoss() const {
        return lossFirst_ / nexamples_;
    }

    real Model::State::getSecondLoss() const {
        return lossSecond_ / nexamples_;
    }

    Model::State::State(int32_t hiddenSize, int32_t outputSize, int thread_id, int32_t seed)
            : lossFirst_(0.0),
              lossSecond_(0.0),
              nexamples_(0),
              inputGrad(hiddenSize),
              inputVec(hiddenSize),
              outputVec(outputSize),
              secOutVec(outputSize),
              sumOutVec(outputSize),
              rng(seed),
              thread_id(thread_id),
              inId(0),
              inNorm(0.0){}

    Model::Model(
            std::shared_ptr<Matrix> wi,
            std::shared_ptr<Matrix> wo,
            std::shared_ptr<Loss> loss,
            bool normalizeGradient)
            : wi_(wi), wo_(wo), loss_(loss), normalizeGradient_(normalizeGradient) {}

} // namespace fasttext
