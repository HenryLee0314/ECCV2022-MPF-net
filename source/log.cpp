#include "log.h"

#ifdef _WIN32
#include <sys/timeb.h>
#include <time.h>
#include <windows.h>
#else
#include <sys/time.h>
#endif

namespace UTILITY {

#ifdef _WIN32

int gettimeofday(struct timeval *tp, void *tzp) {
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year = wtm.wYear - 1900;
    tm.tm_mon = wtm.wMonth - 1;
    tm.tm_mday = wtm.wDay;
    tm.tm_hour = wtm.wHour;
    tm.tm_min = wtm.wMinute;
    tm.tm_sec = wtm.wSecond;
    tm.tm_isdst = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#endif

} // namespace UTILITY

unsigned long us_ticker_read() {
    struct timeval timer;
#ifdef _WIN32
    UTILITY::gettimeofday(&(timer), nullptr);
#else
    gettimeofday(&(timer), nullptr);
#endif
    return timer.tv_sec * 1000000 + timer.tv_usec;
}

