#include "rga_lock.h"

std::mutex& GlobalRgaMutex() {
    static std::mutex m;
    return m;
}