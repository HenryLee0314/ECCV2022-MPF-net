#include "opencl_task.h"

#include "opencl_manager.h"
#include "log.h"

#include <string>
#include <iostream>

namespace CGRA {

OpenclTask::OpenclTask(const char* const fileAddress)
	: program()
{
	/**Step 5: Create program object */
	std::string fileContent = "";

	int size = 1;

	FILE* file = fopen(fileAddress, "r");
    if (file == nullptr) {
        CGRA_LOGE("file == nullptr");
    }
    else {
        CGRA_LOGD("path: %s", fileAddress);
        fseek(file, 0L, SEEK_END);
        size = ftell(file);
        fseek(file, 0L, SEEK_SET);
    }

    char* _data = (char*)malloc((size + 1) * sizeof(uint8_t));
    _data[size] = '\0';
    fread(_data, sizeof(uint8_t), size, file);

	const char *source = _data;
	size_t sourceSize[] = {(size_t)fileContent.size()};
	program = clCreateProgramWithSource(OpenclManager::getInstance()->getContent(), 1, &source, sourceSize, NULL);

	/**Step 6: Build program. */
	uint32_t err = clBuildProgram(program, 1, OpenclManager::getInstance()->getDevices(), NULL, NULL, NULL);
	if (err != 0) {
		size_t logSize;

		clGetProgramBuildInfo(program, *OpenclManager::getInstance()->getDevices(), CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
		char* log = nullptr;
		log = (char*)malloc((logSize+1)*sizeof(char));
		log[logSize] = '\0';
		clGetProgramBuildInfo(program, *OpenclManager::getInstance()->getDevices(), CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

		CGRA_LOGE("LOG:\n%s\n\n", log);
		free(log);
	}
}

void OpenclTask::run() {

}

OpenclTask::~OpenclTask()
{
	clReleaseProgram(program);
}

} // namespace CGRA

