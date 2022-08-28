#ifndef TOOL_LOG_H
#define TOOL_LOG_H

#include <stdio.h>

#ifdef _WIN32
#include "windows.h"

#define getpid() GetCurrentProcessId()
#define gettid() GetCurrentThreadId()

#else

#include <unistd.h> // getpid()

#endif

#if defined(_WIN32)

#define __UTILITY_FILE_NAME__ fileNameParser(__FILE__, sizeof(__FILE__))

constexpr const char* fileNameParser(const char* fileName, int ptr) {
    return (ptr < 0 ? (fileName) : ((fileName[ptr] == '\\' || fileName[ptr] == '/') ? (fileName + ptr + 1) : (fileNameParser(fileName, ptr - 1))));
}

#endif

#if defined(__linux__)

#include <sys/syscall.h> // gettid()
#define gettid() syscall(__NR_gettid)

#define __UTILITY_FILE_NAME__ fileNameParser(__FILE__, sizeof(__FILE__))

constexpr const char* fileNameParser(const char* fileName, int ptr) {
    return (ptr < 0 ? (fileName) : ((fileName[ptr] == '\\' || fileName[ptr] == '/') ? (fileName + ptr + 1) : (fileNameParser(fileName, ptr - 1))));
}

#elif defined(__APPLE__)

#include <thread>
#define gettid() (std::hash<std::thread::id>{}(std::this_thread::get_id()))

#define __UTILITY_FILE_NAME__ __FILE_NAME__

// constexpr const char* fileNameParser(const char* fileName, int ptr) {
//     return (ptr < 0 ? (fileName) : ((fileName[ptr] == '\\' || fileName[ptr] == '/') ? (fileName + ptr) : (fileNameParser(fileName, ptr - 1))));
// }

#endif

#ifndef __UTILITY_FILE_NAME__
#define __UTILITY_FILE_NAME__
#endif

#ifdef __cplusplus
extern "C" {
#endif

unsigned long us_ticker_read();

#ifdef __cplusplus
}
#endif

static unsigned long g_init_time = us_ticker_read();
#define gettime() ((us_ticker_read() - g_init_time) / 1000)

#define UTILITY_LOG_LEVEL_ERROR     "E"
#define UTILITY_LOG_LEVEL_WARNING   "W"
#define UTILITY_LOG_LEVEL_DEBUG     "D"

#define UTILITY_LOG(logLevel, format, ...) printf("%s/[%ld] (%d) <%zu> %s[%d] %s() " format "\n", logLevel, gettime(), getpid(), gettid(), __UTILITY_FILE_NAME__, __LINE__, __func__, ##__VA_ARGS__)

#define UTILITY_LOGE(format, ...) UTILITY_LOG(UTILITY_LOG_LEVEL_ERROR, format, ##__VA_ARGS__)
#define UTILITY_LOGW(format, ...) UTILITY_LOG(UTILITY_LOG_LEVEL_WARNING, format, ##__VA_ARGS__)
#define UTILITY_LOGD(format, ...) UTILITY_LOG(UTILITY_LOG_LEVEL_DEBUG, format, ##__VA_ARGS__)

#define UTILITY_LOGP(format, ...) printf(format "\n",##__VA_ARGS__)



#define CGRA_LOGE(format, ...) UTILITY_LOGE(format, ##__VA_ARGS__)
#define CGRA_LOGW(format, ...) UTILITY_LOGW(format, ##__VA_ARGS__)
#define CGRA_LOGD(format, ...) UTILITY_LOGD(format, ##__VA_ARGS__)

#define CGRA_LOGP(format, ...) UTILITY_LOGP(format "\n",##__VA_ARGS__)

#endif //_TOOL_LOG_H
