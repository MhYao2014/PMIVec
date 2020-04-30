//
// Created by mhyao on 2020/4/24.
//
#include <thread>
#include <iomanip>
#include "SecVec.h"
#include "utility.h"

SecVec::SecVec():dim(0),
                 vocabSize(0),
                 process(0.0),
                 lossFirst(0.0),
                 lossSecond(0.0),
                 allThreadToken(0),
                 p2Input(NULL),
                 p2Output(NULL){}

void SecVec::biLogiFirst(long long outId,
                        GradManager& gradient,
                        bool lableIsPositive,
                        double& tmpLossFirst) {
    double inner = gradient.inputVec.dotMul(gradient.outputVec,1.0);
    double score = sigmoid(inner);
    double alpha = (double(lableIsPositive) - score);
    p2Output->addVectorToRow(gradient.inputVec,outId,gradient.lr * alpha);
    gradient.inputGrad.addVector(gradient.outputVec,-1.0*alpha);// need positive grad of input Vec
    if (lableIsPositive) {
        tmpLossFirst += - log(score);
    } else {
        tmpLossFirst += - log(1.0 - score);
    }
}

void SecVec::biLogiSecond(long long outId,
                        long long secOutId,
                        GradManager& gradient,
                        bool lableIsPositive,
                        double& tmpLossSecond) {
    // summation vector
    gradient.sumOutVec.zero();
    gradient.sumOutVec.addVector(gradient.outputVec,1.0);
    gradient.sumOutVec.addVector(gradient.secOutVec,1.0);
    double inner = gradient.inputVec.dotMul(gradient.sumOutVec,1.0);
    double score = sigmoid(inner);
    double alpha = (double(lableIsPositive) - score);
    p2Output->addVectorToRow(gradient.inputVec,outId, gradient.lr * alpha);
    p2Output->addVectorToRow(gradient.inputVec,secOutId, gradient.lr * alpha);
    gradient.inputGrad.addVector(gradient.sumOutVec,-1.0 * alpha);// need positive grad of input Vec
    if (lableIsPositive) {
        tmpLossSecond += -log(score);
    } else {
        tmpLossSecond += -log(1.0 - score);
    }
}

void SecVec::gradUpdate(GradManager& gradient, Args* p2Args) {
    // Riemannian gradient update
    // I^{(t+1)} = (I^{(t)}+Z) / (||I^{(t)}+Z||)
    // Z = -1 * lr * (1 + <I^{(t)},InGrad>/||InGrad||) * (InGrad - <I^{(t)},InGrad> * I^{(t)})
    double projectScale = gradient.inputVec.dotMul(gradient.inputGrad,1.0);
    Vector Z(p2Args->dim);
    Z.zero();Z.addVector(gradient.inputGrad,1.0);
    Z.addVector(gradient.inputVec,-1.0*projectScale);
    Z.scalerMul(-1.0 * gradient.lr * (1.0+projectScale/gradient.inputGrad.norm()));
    p2Input->addVectorToRow(Z, gradient.inId,1.0);
    p2Input->scalerMulRow(1.0 / p2Input->l2NormRow(gradient.inId), gradient.inId);
}

void SecVec::forward(long long outId,
        long long secOutId,
        GradManager& gradient,
        double& tmpLossFirst,
        double& tmpLossSecond,
        Args* p2Args) {
    long long negOutId,secNegOutId;
    gradient.outputVec.zero();gradient.outputVec.addRow(*p2Output,outId,1.0);
    gradient.secOutVec.zero();gradient.secOutVec.addRow(*p2Output,secOutId,1.0);
    biLogiFirst(outId,gradient,true,tmpLossFirst);
    biLogiSecond(outId,secOutId,gradient,true,tmpLossSecond);
    for (int neg = 0; neg < p2Args->neg; neg++) { // first term's neg part
        negOutId = getNegative(outId,gradient.rng);
        gradient.outputVec.zero();gradient.outputVec.addRow(*p2Output,negOutId,1.0);
        biLogiFirst(negOutId,gradient, false,tmpLossFirst);
    }
    for (int secNegI = 0; secNegI < p2Args->neg % 2; secNegI++) { // second term's neg part
        for (int secNegJ = 0; secNegJ < p2Args->neg % 2; secNegJ++) {
            negOutId = getNegative(outId,gradient.rng);
            secNegOutId = getNegative(secOutId,gradient.rng);
            gradient.outputVec.zero();gradient.outputVec.addRow(*p2Output,negOutId,1.0);
            gradient.secOutVec.zero();gradient.secOutVec.addRow(*p2Output,secNegOutId,1.0);
            biLogiSecond(negOutId,secNegOutId,gradient, false,tmpLossSecond);
        }
    }
    gradient.lossFirst += tmpLossFirst;
    gradient.lossSecond += tmpLossSecond;
    gradient.threadTokenCount += 1;
}

void SecVec::lossEachLine(int threadId,
        Dictionary* p2Dict,
        Args* p2Args,
        std::vector<long long>line,
        GradManager& gradient) {
    long long inId, outId, secOutId;double tmpLossFirst, tmpLossSecond;bool ifUpdate;
    for (int lineIndex = 0; lineIndex < line.size(); lineIndex++) {
        inId = line[lineIndex];ifUpdate = false;
        if (inId != -1) { // skip OOV word
            gradient.inId = inId;
            gradient.inputVec.zero();gradient.inputVec.addRow(*p2Input,inId,1.0);
            gradient.inputGrad.zero();gradient.inNorm = gradient.inputVec.norm();
            gradient.inputVec.scalerMul(1.0/gradient.inNorm);// unit input vector
            for (int shift = -p2Args->ws; shift <= p2Args->ws; shift++) { // lossEachWindow
                if (shift != 0) { // skip inId itself
                    if (lineIndex + shift >= 0 && lineIndex + shift < line.size()) {
                        outId = line[lineIndex + shift];
                        if (outId != -1) { // skip OOV word
                            tmpLossFirst = 0.0; tmpLossSecond = 0.0;
                            ifUpdate = true;
                            secOutId = pickSecOutId(-p2Args->ws,p2Args->ws,outId,lineIndex,shift,line);
                            forward(outId,secOutId,gradient,tmpLossFirst,tmpLossSecond,p2Args);
                        }
                    }
                }
            }
            if (ifUpdate) {
                gradUpdate(gradient,p2Args);
            }
        }
        // only the master thread output its local loss
        if (threadId == 0) {
            lossFirst = gradient.getAverLossFirst();
            lossSecond = gradient.getAverLossSecond();
        }
    }
    // shrink lr, update process and loss
    allThreadToken += line.size(); // each thread add one token each time
    process = shrinkLr(gradient, p2Dict, p2Args);
}

void SecVec::eachThread(int threadId,
        Dictionary* p2Dict,
        Args* p2Args) {
    GradManager gradient(p2Args->dim, threadId);std::vector<long long> line;
    gradient.setLr(p2Args->lr);
    FILE* p2TrainFile = fopen(p2Args->input.c_str(), "r");
    std::fseek(p2TrainFile, threadId * size(p2TrainFile) / p2Args->thread, SEEK_SET);
    for (int epo = 0; epo < p2Args->epoch; epo++) {
        while (IfOneEpoch(p2TrainFile,threadId,p2Args->thread)) {
            p2Dict->getLine(p2TrainFile,line);
            if (!line.empty()) {
                SecVec::lossEachLine(threadId,p2Dict,p2Args,line,gradient);
            }
        }
        std::fseek(p2TrainFile,threadId * size(p2TrainFile) / p2Args->thread, SEEK_SET);
    }
    fclose(p2TrainFile);
}

void SecVec::train(Dictionary *p2Dict, Args *p2Args) {
    initNegAndUniTable(p2Dict);
    std::vector<std::thread> threads;
    process = 0.0;
    // use lambda expressions to excute each thread,
    // while the root thread can still get contact to each thread's variable
    start_ = std::chrono::steady_clock::now();
    for (int i = 0; i < p2Args->thread; i++) {
        threads.push_back(std::thread([=](){eachThread(i,p2Dict,p2Args);}));
    }
    while (process < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cerr << "\r";
        printInfo(process,lossFirst,lossSecond,std::cerr);
    }
    for (int i = 0; i < p2Args->thread; i++) {
        threads[i].join();
    }
    std::cerr << "\r";
    printInfo(1.0,lossFirst,lossSecond,std::cerr);
    std::cerr << std::endl;
    saveVec(p2Dict,p2Args);
}

long long SecVec::pickSecOutId(int lB, int rB,
                               int seed, int lineIndex,int shift,
                               std::vector<long long>& line) {
    std::minstd_rand rng(seed);
    std::uniform_int_distribution<> uniform(lB, rB);
    int secShift;
    while (true) {
        secShift = uniform(rng);
        if (lineIndex+secShift >= 0 && // within left boundary
            lineIndex+secShift < line.size() && // within right boundary
            secShift != shift &&  // skip outId
            line[lineIndex+secShift] != -1) { // skip OOV word
            break;
        }
    }
    return line[lineIndex+secShift];
}

double SecVec::shrinkLr(GradManager& gradient, Dictionary* p2Dict, Args* p2Args) {
    double tmpprocess = (double)(allThreadToken) / (p2Dict->_total_tokens * p2Args->epoch);
    gradient.setLr((1-process)*p2Args->lr);
    return tmpprocess;
}

void SecVec::saveVec(Dictionary* p2Dict,Args* p2Args) {
    FILE* p2VecFile = fopen((p2Args->vecSavePath + \
    "-minCount" + std::to_string(p2Args->minCount) + \
    "-maxVocab" + std::to_string(p2Args->maxVocab) + \
    "-dim" + std::to_string(p2Args->dim) + ".vec").c_str(),"w");
    int64_t index = 0;
    fprintf(p2VecFile,"%lld %d\n",vocabSize,dim);
    for (int64_t i = 0; i < p2Input->rows(); i++) {
        fprintf(p2VecFile,"%s ",p2Dict->vocabArray[i].Word);
        for (int64_t j = 0; j < p2Input->cols(); j++) {
            index = p2Input->cols() * i + j;
            fprintf(p2VecFile, "%f ", p2Input->data()[index]);
        }
        fprintf(p2VecFile, "\n");
    }
    fclose(p2VecFile);
    p2VecFile = fopen((p2Args->vecSavePath + \
    "-minCount" + std::to_string(p2Args->minCount) + \
    "-maxVocab" + std::to_string(p2Args->maxVocab) + \
    "-dim" + std::to_string(p2Args->dim) + ".output").c_str(),"w");
    index = 0;
    fprintf(p2VecFile,"%lld %d\n",vocabSize,dim);
    for (int64_t i = 0; i < p2Output->rows(); i++) {
        fprintf(p2VecFile,"%s ",p2Dict->vocabArray[i].Word);
        for (int64_t j = 0; j < p2Output->cols(); j++) {
            index = p2Output->cols() * i + j;
            fprintf(p2VecFile, "%f ", p2Output->data()[index]);
        }
        fprintf(p2VecFile, "\n");
    }
}

void SecVec::printInfo(double progress,
        double tmplossFirst,
        double tmplossSecond,
        std::ostream& log_stream) {
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double t =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start_)
                    .count();
    double wst = 0; int64_t eta = 2592000; // Default to one month in seconds (720 * 3600)
    progress = progress * 100;
    eta = t * (100 - progress) / progress;
    wst = double(allThreadToken) / t;
    int32_t etah = eta / 3600;
    int32_t etam = (eta % 3600) / 60;

    log_stream << std::fixed;
    log_stream << "Progress: ";
    log_stream << std::setprecision(1) << std::setw(3) << progress << "%";
    log_stream << " lossFirst: " << std::setw(6) << std::setprecision(5) << tmplossFirst;
    log_stream << " lossSecond: " << std::setw(7) << std::setprecision(5) << tmplossSecond;
    log_stream << " words/sec/total: " << std::setw(3) << int64_t(wst);
    log_stream << " ETA: " << std::setw(1) << etah;
    log_stream << "h" << std::setw(2) << etam << "m";
    log_stream << std::flush;
}

void SecVec::initNegAndUniTable(Dictionary* p2Dict) {
    double z = 0.0,c;
    // 遍历哈希链表词典，计算所有的词频加和
    HASHUNITID * htmp = NULL;
    for (long long i = 0 ; i < TSIZE; i++) {
        if (p2Dict->vocabHash[i] != NULL) {
            htmp = p2Dict->vocabHash[i];
            while (htmp != NULL) {
                if (htmp->id != -1) {
                    z += pow(htmp->Count, 0.75);
                }
                htmp = htmp->next;
            }
        }
    }
    // 按照公式，向负采样表里填入相应个数的id
    for (long long i = 0 ; i < TSIZE; i++) {
        if (p2Dict->vocabHash[i] != NULL) {
            htmp = p2Dict->vocabHash[i];
            while (htmp != NULL) {
                if (htmp->id != -1) {
                    c = pow(htmp->Count, 0.75);
                    // 填入对应个数的id
                    for (size_t j = 0; j < c * 10000000 / z; j++) {
                        negatives_.push_back((long long)htmp->id);
                    }
                }
                htmp = htmp->next;
            }
        }
    }
    uniform_ = std::uniform_int_distribution<size_t>(0,negatives_.size() - 1);
}

void SecVec::initVariables(Dictionary *p2Dict, Args *p2Args) {
    vocabSize = p2Dict->getRealVocabSize();
    dim = p2Args->dim;
    p2Input = std::make_shared<Matrix>(vocabSize,dim);
    p2Output = std::make_shared<Matrix>(vocabSize, dim);
    p2Input->uniform(0.05);
    p2Output->uniform(0.05);
}