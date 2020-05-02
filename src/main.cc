/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include "args.h"
#include "fasttext.h"

using namespace fasttext;

void printUsage() {
    std::cerr
            << "usage: fasttext <command> <args>\n\n"
            << "The commands supported by fasttext are:\n\n"
            << "  supervised              train a supervised classifier\n"
            << "  quantize                quantize a model to reduce the memory usage\n"
            << "  test                    evaluate a supervised classifier\n"
            << "  test-label              print labels with precision and recall scores\n"
            << "  predict                 predict most likely labels\n"
            << "  predict-prob            predict most likely labels with probabilities\n"
            << "  skipgram                train a skipgram model\n"
            << "  cbow                    train a cbow model\n"
            << "  print-word-vectors      print word vectors given a trained model\n"
            << "  print-sentence-vectors  print sentence vectors given a trained model\n"
            << "  print-ngrams            print ngrams given a trained model and word\n"
            << "  nn                      query for nearest neighbors\n"
            << "  analogies               query for analogies\n"
            << "  dump                    dump arguments,dictionary,input/output vectors\n"
            << std::endl;
}

void train(const std::vector<std::string> args) {
    Args a = Args();
    a.parseArgs(args);
    std::shared_ptr<FastText> fasttext = std::make_shared<FastText>();
    fasttext->train(a);
    fasttext->saveVectors(a.output + ".vec");
    if (a.saveOutput) {
        fasttext->saveOutput(a.output + ".output");
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 2) {
        printUsage();
        exit(EXIT_FAILURE);
    }
    std::string command(args[1]);
    if (command == "skipgram" || command == "cbow" || command == "supervised") {
        train(args);
    } else {
        printUsage();
        exit(EXIT_FAILURE);
    }
    return 0;
}
