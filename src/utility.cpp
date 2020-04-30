#include "utility.h"
#include <atomic>
#include <unistd.h>

bool IfOneEpoch(FILE *p2File, int threadId, int threadNum){
    // todo:这里写了一个大bug。ftell不能返回有意义的文件位置就很坑。
    // 判断条件为是否到达了下一个线程的起始位置
    long begRegionNext = (threadId + 1) * size(p2File) / threadNum;
    long positionNow = ftell(p2File);
    if (positionNow >= begRegionNext) {
        return false;
    }
    return true;
}

int64_t size(FILE * p2File) {
    long tmpPos = ftell(p2File);
    std::fseek(p2File, 0, SEEK_END);
    long size = ftell(p2File);
    std::fseek(p2File,tmpPos, SEEK_SET);
    return size;
}
