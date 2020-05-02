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
            real lr,
            State& state) {
        computeHidden(input, state);
        Vector& grad = state.grad;
        grad.zero();
        real lossValue = loss_->forward(targets, targetIndex, state, lr, true);
        state.incrementNExamples(lossValue);
        wi_->addVectorToRow(grad, input, 1.0);
    }

    void Model::computeHidden(const int32_t & input, State& state)
    const {
        Vector& hidden = state.hidden;
        hidden.zero();
        hidden.addRow(*wi_, input);
    }

    void Model::State::incrementNExamples(real loss) {
        lossValue_ += loss;
        nexamples_++;
    }

    real Model::State::getLoss() const {
        return lossValue_ / nexamples_;
    }

    Model::State::State(int32_t hiddenSize, int32_t outputSize, int thread_id, int32_t seed)
            : lossValue_(0.0),
              nexamples_(0),
              hidden(hiddenSize),
              output(outputSize),
              grad(hiddenSize),
              rng(seed),
              thread_id(thread_id){}

    Model::Model(
            std::shared_ptr<Matrix> wi,
            std::shared_ptr<Matrix> wo,
            std::shared_ptr<Loss> loss,
            bool normalizeGradient)
            : wi_(wi), wo_(wo), loss_(loss), normalizeGradient_(normalizeGradient) {}

} // namespace fasttext
