#include "gradmanager.h"

void GradManager::setLr(double newLr) {
    lr = newLr;
}

double GradManager::getAverLossFirst() {
    return (double)lossFirst / threadTokenCount;
}

double GradManager::getAverLossSecond() {
    return (double)lossSecond / threadTokenCount;
}

GradManager::GradManager(int dim, int seed):rng(seed),
                                            inId(0),
                                            secOutId(0),
                                            inputGrad(dim),
                                            inputVec(dim),
                                            outputVec(dim),
                                            secOutVec(dim),
                                            sumOutVec(dim),
                                            lossFirst(0.0),
                                            lossSecond(0.0),
                                            threadTokenCount(0),
                                            lr(0.0),
                                            inNorm(0.0){}