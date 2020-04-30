//
// Created by mhyao on 20-1-10.
//
#pragma once
#include "args.h"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <stdexcept>

Args::Args() {
    ifSaveVocab = 1;
    thread = 1;
    minCount = 100;
    maxVocab = 100000;
    lr = 0.05;
    dim = 100;
    ws = 5;
    epoch = 5;
    neg = 5;
}

void Args::printHelp() {
    std::cerr << "\nThe following arguments are mandatory:\n"
              << "  -input              training file path\n"
              << "  -loss               supported loss model lists are: [skip-gram-mpi-openmp, skip-gram-subPro]\n"
              << "  -thread  thread used to split corpus\n"
              << "  -minCount           minimal number of word occurences ["
              << minCount << "]\n"
              << "  -maxVocab           maximal number of word vocabulary ["
              << maxVocab << "]\n"
              << "  -vocabSavePath      where you save your vocabulary\n"
              << "\nThe following arguments for training are optional:\n"
              << "  -lr                 learning rate [" << lr << "]\n"
              << "  -dim                size of word vectors [" << dim << "]\n"
              << "  -ws                 size of the context window [" << ws << "]\n"
              << "  -epoch              number of epochs [" << epoch << "]\n"
              << "  -neg                number of negatives sampled [" << neg << "]\n"
              << "  -vecSavePath        where you save your vectors\n";
}

bool Args::parseArgs(int argc, const std::vector<std::string> &args) {
    // 如果用户没有输入任何参数，那么就返回错误值false
    if (argc <= 1) {
        Args::printHelp();
        return false;
    }
    // 输入了一些参数
    for (int ai = 1; ai < args.size(); ai += 2) {
        if (args[ai][0] != '-') {
            std::cout << "Provided argument without a dash! Usage:" << std::endl;
            Args::printHelp();
            return false;
        }
        try {
            if (args[ai] == "-h") {
                std::cerr << "Here is the help! Usage:" << std::endl;
                Args::printHelp();
                return false;
            } else if (args[ai] == "-input") {
                input = std::string(args.at(ai + 1));
            } else if (args[ai] == "-vocabPath") {
                vocabPath = std::string(args.at(ai + 1));
            } else if (args[ai] == "-loss") {
                loss = std::string(args.at(ai + 1));
            } else if (args[ai] == "-IfSaveVocab") {
                ifSaveVocab = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-corpusSplitThread") {
                thread = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-minCount") {
                minCount = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-maxVocab") {
                maxVocab = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-lr") {
                lr = std::stof(args.at(ai + 1));
            } else if (args[ai] == "-dim") {
                dim = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-ws") {
                ws = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-epoch") {
                epoch = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-neg") {
                neg = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-vecSavePath") {
                vecSavePath = std::string(args.at(ai + 1));
            }
        } catch (std::out_of_range) {
            std::cerr << args[ai] << " is missing an argument" << std::endl;
            Args::printHelp();
            return false;
        }
    }
    return true;
}