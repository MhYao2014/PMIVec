/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fasttext.h"
#include "loss.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fasttext {

    constexpr int32_t FASTTEXT_VERSION = 12; /* Version 1b */
    constexpr int32_t FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;

    void FastText::train(const Args& args) {
        args_ = std::make_shared<Args>(args);
        dict_ = std::make_shared<Dictionary>(args_);
        if (args_->input == "-") {
            // manage expectations
            throw std::invalid_argument("Cannot use stdin for training!");
        }
        std::ifstream ifs(args_->input);
        if (!ifs.is_open()) {
            throw std::invalid_argument(
                    args_->input + " cannot be opened for training!");
        }
        dict_->readFromFile(ifs);
        ifs.close();
        input_ = createRandomMatrix();
        output_ = createTrainOutputMatrix();
        auto loss = createLoss(output_);
        bool normalizeGradient = (args_->model == model_name::sup);
        model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
        startThreads();
    }

    void FastText::startThreads() {
        start_ = std::chrono::steady_clock::now();
        tokenCount_ = 0;
        loss_ = -1;
        trainException_ = nullptr;
        std::vector<std::thread> threads;
        for (int32_t i = 0; i < args_->thread; i++) {
            threads.push_back(std::thread([=]() { trainThread(i); }));
        }
        const int64_t ntokens = dict_->ntokens();
        // Same condition as trainThread
        while (keepTraining(ntokens)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (loss_ >= 0 && args_->verbose > 1) {
                real progress = real(tokenCount_) / (args_->epoch * ntokens);
                std::cerr << "\r";
                printInfo(progress, loss_, std::cerr);
            }
        }
        for (int32_t i = 0; i < args_->thread; i++) {
            threads[i].join();
        }
        if (trainException_) {
            std::exception_ptr exception = trainException_;
            trainException_ = nullptr;
            std::rethrow_exception(exception);
        }
        if (args_->verbose > 0) {
            std::cerr << "\r";
            printInfo(1.0, loss_, std::cerr);
            std::cerr << std::endl;
        }
    }

    void FastText::trainThread(int32_t threadId) {
        std::ifstream ifs(args_->input);
        utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

        Model::State state(args_->dim, output_->size(0), threadId, threadId + args_->seed);

        const int64_t ntokens = dict_->ntokens();
        int64_t localTokenCount = 0;
        std::vector<int32_t> line, labels;
        try {
            while (keepTraining(ntokens)) {
                real progress = real(tokenCount_) / (args_->epoch * ntokens);
                real lr = args_->lr * (1.0 - progress);
                localTokenCount += dict_->getLine(ifs, line, state.rng);
                skipgram(state, lr, line);
                if (localTokenCount > args_->lrUpdateRate) {
                    tokenCount_ += localTokenCount;
                    localTokenCount = 0;
                    if (threadId == 0 && args_->verbose > 1) {
                        loss_ = state.getLoss();
                    }
                }
            }
        } catch (DenseMatrix::EncounteredNaNError&) {
            trainException_ = std::current_exception();
        }
        if (threadId == 0)
            loss_ = state.getLoss();
        ifs.close();
    }

    bool FastText::keepTraining(const int64_t ntokens) const {
        return tokenCount_ < args_->epoch * ntokens && !trainException_;
    }

    void FastText::skipgram(
            Model::State& state,
            real lr,
            const std::vector<int32_t>& line) {
        std::uniform_int_distribution<> uniform(1, args_->ws);
        for (int32_t w = 0; w < line.size(); w++) {
            int32_t boundary = uniform(state.rng);
            const int32_t & inDictId = line[w];
            for (int32_t c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    model_->update(inDictId, line, w + c, lr, state);
                }
            }
        }
    }

    void FastText::printInfo(real progress, real loss, std::ostream& log_stream) {
        double t = utils::getDuration(start_, std::chrono::steady_clock::now());
        double lr = args_->lr * (1.0 - progress);
        double wst = 0;

        int64_t eta = 2592000; // Default to one month in seconds (720 * 3600)

        if (progress > 0 && t >= 0) {
            progress = progress * 100;
            eta = t * (100 - progress) / progress;
            wst = double(tokenCount_) / t / args_->thread;
        }

        log_stream << std::fixed;
        log_stream << "Progress: ";
        log_stream << std::setprecision(1) << std::setw(5) << progress << "%";
        log_stream << " words/sec/thread: " << std::setw(7) << int64_t(wst);
        log_stream << " lr: " << std::setw(9) << std::setprecision(6) << lr;
        log_stream << " avg.loss: " << std::setw(9) << std::setprecision(6) << loss;
        log_stream << " ETA: " << utils::ClockPrint(eta);
        log_stream << std::flush;
    }

    std::shared_ptr<Matrix> FastText::createRandomMatrix() const {
        std::shared_ptr<DenseMatrix> input = std::make_shared<DenseMatrix>(
                dict_->nwords() + args_->bucket, args_->dim);
        input->uniform(1.0 / args_->dim, args_->thread, args_->seed);

        return input;
    }

    std::shared_ptr<Matrix> FastText::createTrainOutputMatrix() const {
        int64_t m = dict_->nwords();
        std::shared_ptr<DenseMatrix> output =
                std::make_shared<DenseMatrix>(m, args_->dim);
        output->zero();

        return output;
    }

    void FastText::getWordVector(Vector& vec, const std::string& word) const {
        const int32_t & dictId = dict_->getId(word);
        vec.zero();
        addInputVector(vec, dictId);
    }

    void FastText::saveVectors(const std::string& filename) {
        if (!input_ || !output_) {
            throw std::runtime_error("Model never trained");
        }
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            throw std::invalid_argument(
                    filename + " cannot be opened for saving vectors!");
        }
        ofs << dict_->nwords() << " " << args_->dim << std::endl;
        Vector vec(args_->dim);
        for (int32_t i = 0; i < dict_->nwords(); i++) {
            std::string word = dict_->getWord(i);
            getWordVector(vec, word);
            ofs << word << " " << vec << std::endl;
        }
        ofs.close();
    }

    void FastText::saveOutput(const std::string& filename) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            throw std::invalid_argument(
                    filename + " cannot be opened for saving vectors!");
        }
        int32_t n = dict_->nwords();
        ofs << n << " " << args_->dim << std::endl;
        Vector vec(args_->dim);
        for (int32_t i = 0; i < n; i++) {
            std::string word = dict_->getWord(i);
            vec.zero();
            vec.addRow(*output_, i);
            ofs << word << " " << vec << std::endl;
        }
        ofs.close();
    }

    void FastText::addInputVector(Vector& vec, int32_t ind) const {
        vec.addRow(*input_, ind);
    }


    std::shared_ptr<Loss> FastText::createLoss(std::shared_ptr<Matrix>& output) {
        loss_name lossName = args_->loss;
        switch (lossName) {
            case loss_name::ns:
                return std::make_shared<NegativeSamplingLoss>(
                        output, args_->neg, getTargetCounts());
            default:
                throw std::runtime_error("Unknown loss");
        }
    }

    std::vector<int64_t> FastText::getTargetCounts() const {
        if (args_->model == model_name::sup) {
            return dict_->getCounts(entry_type::label);
        } else {
            return dict_->getCounts(entry_type::word);
        }
    }

    FastText::FastText()
            : wordVectors_(nullptr), trainException_(nullptr) {}

} // namespace fasttext
