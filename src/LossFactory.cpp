#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "LossFactory.h"
#include "SecVec.h"

LossFactory::LossFactory(): _lib_can_load(false) {}

void LossFactory::setLossName(std::string & loss_name) {
    _loss_name = loss_name;
}

Loss* LossFactory::createLoss() {
    if (_loss_name == "SecVec") {
        // 在堆上创建具体化一个匿民对象
        // 并用一个对象指针指向这个对象在堆上的地址
        auto* p2myloss = new SecVec;
        return p2myloss;
    }
}