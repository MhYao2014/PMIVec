/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dictionary.h"

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace fasttext {

    const std::string Dictionary::EOS = "</s>";

    void Dictionary::readFromFile(std::istream& in) {
        std::string word;
        int64_t minThreshold = 1;
        while (readWord(in, word)) {
            add(word);
            if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
                std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
            }
            if (size_ > 0.75 * MAX_VOCAB_SIZE) {
                minThreshold++;
                threshold(minThreshold, minThreshold);
            }
        }
        threshold(args_->minCount, args_->minCountLabel);
        initTableDiscard();
        if (args_->verbose > 0) {
            std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
            std::cerr << "Number of words:  " << nwords_ << std::endl;
        }
        if (size_ == 0) {
            throw std::invalid_argument(
                    "Empty vocabulary. Try a smaller -minCount value.");
        }
    }

    bool Dictionary::readWord(std::istream& in, std::string& word) const {
        int c;
        std::streambuf& sb = *in.rdbuf();
        word.clear();
        while ((c = sb.sbumpc()) != EOF) {
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
                c == '\f' || c == '\0') {
                if (word.empty()) {
                    if (c == '\n') {
                        word += EOS;
                        return true;
                    }
                    continue;
                } else {
                    if (c == '\n')
                        sb.sungetc();
                    return true;
                }
            }
            word.push_back(c);
        }
        // trigger eofbit
        in.get();
        return !word.empty();
    }

    void Dictionary::add(const std::string& w) {
        int32_t h = find(w);
        ntokens_++;
        if (word2int_[h] == -1) {
            entry e;
            e.word = w;
            e.count = 1;
            e.dictId = size_;
            words_.push_back(e);
            word2int_[h] = size_++;
        } else {
            words_[word2int_[h]].count++;
            words_[word2int_[h]].dictId = word2int_[h];
        }
    }

    void Dictionary::threshold(int64_t t, int64_t tl) {
        sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
            return e1.count > e2.count;
        });
        words_.erase(
                remove_if(
                        words_.begin(),
                        words_.end(),
                        [&](const entry& e) {
                            return (e.count < t) ||
                                   (e.count < tl);
                        }),
                words_.end());
        words_.shrink_to_fit();
        size_ = 0;
        nwords_ = 0;
        std::fill(word2int_.begin(), word2int_.end(), -1);
        for (auto it = words_.begin(); it != words_.end(); ++it) {
            int32_t h = find(it->word);
            it->dictId = size_;
            word2int_[h] = size_++;
            nwords_++;
        }
    }

    void Dictionary::initTableDiscard() {
        pdiscard_.resize(size_);
        for (size_t i = 0; i < size_; i++) {
            real f = real(words_[i].count) / real(ntokens_);
            pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
        }
    }

    int32_t Dictionary::getLine(
            std::istream& in,
            std::vector<int32_t>& words,
            std::minstd_rand& rng) const {
        std::uniform_real_distribution<> uniform(0, 1);
        std::string token;
        int32_t ntokens = 0;

        reset(in);
        words.clear();
        while (readWord(in, token)) {
            int32_t h = find(token);
            int32_t wid = word2int_[h];
            if (wid < 0) {
                continue;
            }

            ntokens++;
            if (!discard(wid, uniform(rng))) {
                words.push_back(wid);
            }
            if (ntokens > MAX_LINE_SIZE || token == EOS) {
                break;
            }
        }
        return ntokens;
    }

    bool Dictionary::discard(int32_t id, real rand) const {
        assert(id >= 0);
        assert(id < nwords_);
        if (args_->model == model_name::sup) {
            return false;
        }
        return rand > pdiscard_[id];
    }

    int32_t Dictionary::find(const std::string& w) const {
        return find(w, hash(w));
    }

    void Dictionary::reset(std::istream& in) const {
        if (in.eof()) {
            in.clear();
            in.seekg(std::streampos(0));
        }
    }

    int32_t Dictionary::getId(const std::string& w) const {
        int32_t h = find(w);
        return word2int_[h];
    }

    std::string Dictionary::getWord(int32_t id) const {
        assert(id >= 0);
        assert(id < size_);
        return words_[id].word;
    }

// The correct implementation of fnv should be:
// h = h ^ uint32_t(uint8_t(str[i]));
// Unfortunately, earlier version of fasttext used
// h = h ^ uint32_t(str[i]);
// which is undefined behavior (as char can be signed or unsigned).
// Since all fasttext models that were already released were trained
// using signed char, we fixed the hash function to make models
// compatible whatever compiler is used.
    uint32_t Dictionary::hash(const std::string& str) const {
        uint32_t h = 2166136261;
        for (size_t i = 0; i < str.size(); i++) {
            h = h ^ uint32_t(int8_t(str[i]));
            h = h * 16777619;
        }
        return h;
    }

    std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
        std::vector<int64_t> counts;
        for (auto& w : words_) {
            counts.push_back(w.count);
        }
        return counts;
    }

    Dictionary::Dictionary(std::shared_ptr<Args> args)
            : args_(args),
              word2int_(MAX_VOCAB_SIZE, -1),
              size_(0),
              nwords_(0),
              ntokens_(0) {}

    int32_t Dictionary::find(const std::string& w, uint32_t h) const {
        int32_t word2intsize = word2int_.size();
        int32_t id = h % word2intsize;
        while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
            id = (id + 1) % word2intsize;
        }
        return id;
    }

    int32_t Dictionary::nwords() const {
        return nwords_;
    }

    int64_t Dictionary::ntokens() const {
        return ntokens_;
    }

} // namespace fasttext