//
// Created by mhyao on 2020/4/16.
//

#pragma once
#include <iostream>
#include "args.h"
#include "dictionary.h"

bool IfOneEpoch(FILE *p2File, int threadId, int threadNum);

int64_t size(FILE * p2File);