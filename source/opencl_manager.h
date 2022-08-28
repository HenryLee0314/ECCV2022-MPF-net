#ifndef OPENCL_MANAGER_H
#define OPENCL_MANAGER_H

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace CGRA {

class OpenclManager
{
public:
	static OpenclManager* getInstance();

	cl_context getContent() {
		return context;
	}

	cl_device_id* getDevices() {
		return devices;
	}

	cl_command_queue getCommandQueue() {
		return commandQueue;
	}

private:
	OpenclManager();
	virtual ~OpenclManager();
	OpenclManager(const OpenclManager&);
	OpenclManager& operator = (const OpenclManager&);

	static OpenclManager* _instance;

private:
	cl_platform_id platform;
	cl_device_id* devices;
	cl_context context;
	cl_command_queue commandQueue;
};

} // namespace CGRA

#endif // OPENCL_MANAGER_H