#include <iostream>
#include <vector>
#include "args.h"
#include "dictionary.h"
#include "LossFactory.h"
#include "Loss.h"

int main(int argc, char**argv) {
    std::vector<std::string> args(argv,argv + argc);
    auto arg = Args();
    arg.parseArgs(argc,args);
    auto dict = Dictionary();
    dict.setCorpusPath(arg.input);
    dict.setMinCount(arg.minCount);
    dict.setMaxVocab(arg.maxVocab);
    dict.setIfSaveVocab(arg.ifSaveVocab);
    dict.buildVocab();
    LossFactory lossFactory = LossFactory();
    lossFactory.setLossName(arg.loss);
    Loss* p2MyLoss = lossFactory.createLoss();
    p2MyLoss->initVariables(&dict,&arg);
    p2MyLoss->train(&dict,&arg);
    return 0;
}