#ifndef OPENCL_TASK_H
#define OPENCL_TASK_H

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace CGRA {

class OpenclTask
{
public:
	OpenclTask(const char* const fileAddress);

	virtual ~OpenclTask();

	virtual void run();

protected:
	cl_program program;
};

} // namespace CGRA

#endif // OPENCL_TASK_H