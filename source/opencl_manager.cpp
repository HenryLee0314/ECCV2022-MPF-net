#include "opencl_manager.h"

#include <stdio.h>
#include <stdlib.h>

#include "log.h"

namespace CGRA {

OpenclManager* OpenclManager::_instance = nullptr;

OpenclManager* OpenclManager::getInstance()
{
	if (_instance == nullptr) {
		_instance = new OpenclManager();
	}

	return _instance;
}

OpenclManager::OpenclManager()
	: platform()
	, devices(nullptr)
	, context()
	, commandQueue()
{
	/**Step 1: Getting platforms and choose an available one(first).*/
	cl_uint numPlatforms = 0;
	if (clGetPlatformIDs(0, NULL, &numPlatforms) != CL_SUCCESS)
	{
		CGRA_LOGE();
	}

	/**For clarity, choose the first available platform. */
	if (numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms * sizeof(cl_platform_id));
		clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];

		free(platforms);
	}

	/**Step 2:Query the platform and choose the first GPU device if has one.*/
	cl_uint numDevices = 0;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if (numDevices > 0) //GPU available.
	{
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}

	/**Step 3: Create context.*/
	context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);

	/**Step 4: Creating command queue associate with the context.*/
	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

}


OpenclManager::~OpenclManager()
{
	/**Step 12: Clean the resources.*/
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(context);

	if (devices != NULL)
	{
		free(devices);
		devices = NULL;
	}
}

} // namespace CGRA