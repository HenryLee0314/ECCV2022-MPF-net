
#include "cubemap_top.h"
#include "log.h"
#include "opencl_manager.h"
#include "opencl_task.h"

#include <opencv2/opencv.hpp>

#include "cmake_auto_generate_config.h"

struct FlowData
{
    float x;
    float y;
};

using namespace CGRA;

Cube2Cyli *_proj = new Cube2Cyli(256, 45);


class Equirect2Cylinder : public OpenclTask
{
public:
    Equirect2Cylinder() : OpenclTask(CMAKE_OPENCL_PROGRAM_PATH)
    {
        k_RGB = clCreateKernel(program, "RGB", NULL);
        k_flow = clCreateKernel(program, "flow", NULL);

        k_e2p_RGB = clCreateKernel(program, "e2p_RGB", NULL);

        k_e2p_flow = clCreateKernel(program, "e2p_flow", NULL);

        _image_width = _proj->_image_width;
        _image_height = _proj->_image_height;

        _cl_mem_front = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, _proj->_output_width * _proj->_output_height * sizeof(CubemapsFaceCoordThetaPhi), (void *)_proj->cyli_front, NULL);
        _cl_mem_top = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, _proj->_output_width * _proj->_output_height * sizeof(CubemapsFaceCoordThetaPhi), (void *)_proj->cyli_top, NULL);
        _cl_mem_left = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, _proj->_output_width * _proj->_output_height * sizeof(CubemapsFaceCoordThetaPhi), (void *)_proj->cyli_left, NULL);

        cyli_buff = (uint8_t*)malloc(3 * _proj->getOutputImageHeight() * _proj->getOutputImageWidth() * 3 * sizeof(uint8_t));

        _cl_mem_cyli_buff = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * _proj->getOutputImageHeight() * _proj->getOutputImageWidth() * 3 * sizeof(uint8_t), (void *)cyli_buff, NULL);
        single_height = _proj->getOutputImageHeight();

        cyli_flow_buff = (FlowData *)malloc(3 * _proj->getOutputImageHeight() * _proj->getOutputImageWidth() * sizeof(FlowData));
        _cl_mem_flow_output = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * _proj->getOutputImageHeight() * _proj->getOutputImageWidth() * sizeof(FlowData), (void *)cyli_flow_buff, NULL);
    
        cylinder_rows = 3 * _proj->getOutputImageHeight();
        cylinder_cols = _proj->getOutputImageWidth();
    
    
        //------------------
        // output_RGB_front = cv::Mat(256, 256, CV_8UC3, cv::Scalar(0, 0, 0));
        // output_RGB_back = cv::Mat(256, 256, CV_8UC3, cv::Scalar(0, 0, 0));
        // output_RGB_left = cv::Mat(256, 256, CV_8UC3, cv::Scalar(0, 0, 0));
        // output_RGB_right = cv::Mat(256, 256, CV_8UC3, cv::Scalar(0, 0, 0));
        // output_RGB_top = cv::Mat(256, 256, CV_8UC3, cv::Scalar(0, 0, 0));
        // output_RGB_bottom = cv::Mat(256, 256, CV_8UC3, cv::Scalar(0, 0, 0));

        
        output_RGB_data_front = (uint8_t*)malloc(3 * 256 * 256 * sizeof(uint8_t));
        output_RGB_data_back = (uint8_t*)malloc(3 * 256 * 256 * sizeof(uint8_t));
        output_RGB_data_left = (uint8_t*)malloc(3 * 256 * 256 * sizeof(uint8_t));
        output_RGB_data_right = (uint8_t*)malloc(3 * 256 * 256 * sizeof(uint8_t));
        output_RGB_data_top = (uint8_t*)malloc(3 * 256 * 256 * sizeof(uint8_t));
        output_RGB_data_bottom = (uint8_t*)malloc(3 * 256 * 256 * sizeof(uint8_t));

        _cl_mem_output_RGB_data_front = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * 256 * 256 * sizeof(uint8_t),
                                                       (void *)output_RGB_data_front, NULL);
        _cl_mem_output_RGB_data_back = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * 256 * 256 * sizeof(uint8_t),
                                                      (void *)output_RGB_data_back, NULL);
        _cl_mem_output_RGB_data_left = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * 256 * 256 * sizeof(uint8_t),
                                                      (void *)output_RGB_data_left, NULL);
        _cl_mem_output_RGB_data_right = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * 256 * 256 * sizeof(uint8_t),
                                                       (void *)output_RGB_data_right, NULL);
        _cl_mem_output_RGB_data_top = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * 256 * 256 * sizeof(uint8_t),
                                                     (void *)output_RGB_data_top, NULL);
        _cl_mem_output_RGB_data_bottom = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * 256 * 256 * sizeof(uint8_t),
                                                        (void *)output_RGB_data_bottom, NULL);

        output_flow_front = (FlowData *)malloc(256 * 256 * sizeof(FlowData));
        output_flow_back = (FlowData *)malloc(256 * 256 * sizeof(FlowData));
        output_flow_left = (FlowData *)malloc(256 * 256 * sizeof(FlowData));
        output_flow_right = (FlowData *)malloc(256 * 256 * sizeof(FlowData));
        output_flow_top = (FlowData *)malloc(256 * 256 * sizeof(FlowData));
        output_flow_bottom = (FlowData *)malloc(256 * 256 * sizeof(FlowData));

        _cl_mem_output_flow_front = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 256 * 256 * sizeof(FlowData),
                                                          (void *)output_flow_front, NULL);
        _cl_mem_output_flow_back = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 256 * 256 * sizeof(FlowData),
                                                         (void *)output_flow_back, NULL);
        _cl_mem_output_flow_left = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 256 * 256 * sizeof(FlowData),
                                                         (void *)output_flow_left, NULL);
        _cl_mem_output_flow_right = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 256 * 256 * sizeof(FlowData),
                                                          (void *)output_flow_right, NULL);
        _cl_mem_output_flow_top = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 256 * 256 * sizeof(FlowData),
                                                        (void *)output_flow_top, NULL);
        _cl_mem_output_flow_bottom = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 256 * 256 * sizeof(FlowData),
                                                           (void *)output_flow_bottom, NULL);
    }

    // out set
    FlowData *input_flow_equirect;

    // full cylinder
    uint32_t _image_width;
    uint32_t _image_height;
    int single_height;// = _proj->getOutputImageHeight();
    uint32_t cylinder_rows;
    uint32_t cylinder_cols;

    cv::Mat cyli;
    uint8_t* cyli_buff;

    FlowData *cyli_flow_buff;

    cl_mem _cl_mem_front;
    cl_mem _cl_mem_top;
    cl_mem _cl_mem_left;
    cl_mem _cl_mem_input_buff;
    cl_mem _cl_mem_cyli_buff;

    cl_mem _cl_mem_flow_input;
    cl_mem _cl_mem_flow_output;

    cl_kernel k_RGB;
    cl_kernel k_flow;

    cl_kernel k_e2p_RGB;
    cl_kernel k_e2p_flow;

    cl_event *_enentPoint;

    ~Equirect2Cylinder()
    {
        free(cyli_buff);

        clReleaseKernel(k_RGB);

        clReleaseMemObject(_cl_mem_front);
        clReleaseMemObject(_cl_mem_top);
        clReleaseMemObject(_cl_mem_left);
        clReleaseMemObject(_cl_mem_cyli_buff);
        clReleaseMemObject(_cl_mem_flow_output);
    }

    void init_input(uint8_t* data)
    {
        _cl_mem_input_buff = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 512 * 1024 * 3 * sizeof(uint8_t), (void *)data, NULL);
    }

    void init_flow_input(FlowData *flow)
    {
        _cl_mem_flow_input = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 512 * 1024 * sizeof(FlowData), (void *)flow, NULL);
    }

    void release_RGB_input()
    {
        clReleaseMemObject(_cl_mem_input_buff);
    }

    void release_flow_input()
    {
        clReleaseMemObject(_cl_mem_flow_input);
    }


    void init_input(cv::Mat input_RGB_equirect, FlowData *flow)
    {
        _cl_mem_input_buff = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 512 * 1024 * 3 * sizeof(uint8_t), (void *)input_RGB_equirect.data, NULL);
        _cl_mem_flow_input = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 512 * 1024 * sizeof(FlowData), (void *)flow, NULL);
    }

    void release_input()
    {
        clReleaseMemObject(_cl_mem_input_buff);
        clReleaseMemObject(_cl_mem_flow_input);
    }

    void generate_cylin_RGB()
    {
        clSetKernelArg(k_RGB, 0, sizeof(cl_mem), (void *)&_cl_mem_front);
        clSetKernelArg(k_RGB, 1, sizeof(cl_mem), (void *)&_cl_mem_top);
        clSetKernelArg(k_RGB, 2, sizeof(cl_mem), (void *)&_cl_mem_left);
        clSetKernelArg(k_RGB, 3, sizeof(cl_mem), (void *)&_cl_mem_cyli_buff); // output
        clSetKernelArg(k_RGB, 4, sizeof(cl_mem), (void *)&_cl_mem_input_buff); // input
        clSetKernelArg(k_RGB, 5, sizeof(int), (void *)&_image_width); // width
        clSetKernelArg(k_RGB, 6, sizeof(int), (void *)&single_height); // single_height

        /**Step 10: Running the kernel.*/
        size_t global_work_size[1] = {static_cast<size_t>(_image_width * _proj->getOutputImageHeight() * 3)};

        _enentPoint = (cl_event *)malloc(1 * sizeof(cl_event));

        clEnqueueNDRangeKernel(OpenclManager::getInstance()->getCommandQueue(), k_RGB, 1, NULL, global_work_size, NULL, 0, NULL, _enentPoint);

        clEnqueueWaitForEvents(OpenclManager::getInstance()->getCommandQueue(), 1, _enentPoint);

        clWaitForEvents(1, _enentPoint);
        clReleaseEvent(_enentPoint[0]);
        free(_enentPoint);

        // *Step 11: Read the cout put back to host memory.
        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_cyli_buff, CL_TRUE, 0, 3 * _proj->getOutputImageHeight() * _proj->getOutputImageWidth() * 3 * sizeof(uint8_t), cyli_buff, 0, NULL, NULL);

        
        cyli = cv::Mat(_proj->getOutputImageHeight() * 3, _image_width, CV_8UC3, cyli_buff);
        // cv::imshow("cl_RGB_result", cyli);

        

    }

    void synthesizeCyli() {
        clSetKernelArg(k_flow, 0, sizeof(cl_mem), (void *)&_cl_mem_front);
        clSetKernelArg(k_flow, 1, sizeof(cl_mem), (void *)&_cl_mem_top);
        clSetKernelArg(k_flow, 2, sizeof(cl_mem), (void *)&_cl_mem_left);
        clSetKernelArg(k_flow, 3, sizeof(cl_mem), (void *)&_cl_mem_flow_output); // output
        clSetKernelArg(k_flow, 4, sizeof(cl_mem), (void *)&_cl_mem_flow_input); // input
        clSetKernelArg(k_flow, 5, sizeof(int), (void *)&_image_width);          // width
        clSetKernelArg(k_flow, 6, sizeof(int), (void *)&single_height);         // single_height

        /**Step 10: Running the kernel.*/
        size_t global_work_size[1] = {static_cast<size_t>(_image_width * _proj->getOutputImageHeight() * 3)};

        _enentPoint = (cl_event *)malloc(1 * sizeof(cl_event));

        clEnqueueNDRangeKernel(OpenclManager::getInstance()->getCommandQueue(), k_flow, 1, NULL, global_work_size, NULL, 0, NULL, _enentPoint);

        clEnqueueWaitForEvents(OpenclManager::getInstance()->getCommandQueue(), 1, _enentPoint);

        clWaitForEvents(1, _enentPoint);
        clReleaseEvent(_enentPoint[0]);
        free(_enentPoint);

        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_flow_output, CL_TRUE, 0, 3 * _proj->getOutputImageHeight() * _proj->getOutputImageWidth() * sizeof(FlowData), cyli_flow_buff, 0, NULL, NULL);

        // vis_C_optical_flow(cyli, cyli_flow_buff, "opencl");
    }

    void project_RGB_equirect_to_cubemap()
    {
        size_t global_work_size[1] = {static_cast<size_t>(256 * 256)};

        // top
        clSetKernelArg(k_e2p_RGB, 0, sizeof(cl_mem), (void *)&_cl_mem_input_buff);
        clSetKernelArg(k_e2p_RGB, 1, sizeof(cl_mem), (void *)&_cl_mem_output_RGB_data_top);
        clSetKernelArg(k_e2p_RGB, 2, sizeof(cl_mem), (void *)&_cl_mem_output_RGB_data_left);
        clSetKernelArg(k_e2p_RGB, 3, sizeof(cl_mem), (void *)&_cl_mem_output_RGB_data_front);
        clSetKernelArg(k_e2p_RGB, 4, sizeof(cl_mem), (void *)&_cl_mem_output_RGB_data_right);
        clSetKernelArg(k_e2p_RGB, 5, sizeof(cl_mem), (void *)&_cl_mem_output_RGB_data_back);
        clSetKernelArg(k_e2p_RGB, 6, sizeof(cl_mem), (void *)&_cl_mem_output_RGB_data_bottom);

        _enentPoint = (cl_event *)malloc(1 * sizeof(cl_event));

        clEnqueueNDRangeKernel(OpenclManager::getInstance()->getCommandQueue(), k_e2p_RGB, 1, NULL, global_work_size, NULL, 0, NULL, _enentPoint);

        clEnqueueWaitForEvents(OpenclManager::getInstance()->getCommandQueue(), 1, _enentPoint);

        clWaitForEvents(1, _enentPoint);
        clReleaseEvent(_enentPoint[0]);
        free(_enentPoint);

        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_RGB_data_top, CL_TRUE, 0, 3 * 256 * 256 * sizeof(uint8_t), output_RGB_data_top, 0, NULL, NULL);
        output_RGB_top = cv::Mat(256, 256, CV_8UC3, output_RGB_data_top);

        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_RGB_data_left, CL_TRUE, 0, 3 * 256 * 256 * sizeof(uint8_t), output_RGB_data_left, 0, NULL, NULL);
        output_RGB_left = cv::Mat(256, 256, CV_8UC3, output_RGB_data_left);
        
        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_RGB_data_front, CL_TRUE, 0, 3 * 256 * 256 * sizeof(uint8_t), output_RGB_data_front, 0, NULL, NULL);
        output_RGB_front = cv::Mat(256, 256, CV_8UC3, output_RGB_data_front);

        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_RGB_data_right, CL_TRUE, 0, 3 * 256 * 256 * sizeof(uint8_t), output_RGB_data_right, 0, NULL, NULL);
        output_RGB_right = cv::Mat(256, 256, CV_8UC3, output_RGB_data_right);
       
        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_RGB_data_back, CL_TRUE, 0, 3 * 256 * 256 * sizeof(uint8_t), output_RGB_data_back, 0, NULL, NULL);
        output_RGB_back = cv::Mat(256, 256, CV_8UC3, output_RGB_data_back);
        
        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_RGB_data_bottom, CL_TRUE, 0, 3 * 256 * 256 * sizeof(uint8_t), output_RGB_data_bottom, 0, NULL, NULL);
        output_RGB_bottom = cv::Mat(256, 256, CV_8UC3, output_RGB_data_bottom);
    }

    cv::Mat output_RGB_front;
    uint8_t* output_RGB_data_front;
    cv::Mat output_RGB_back;
    uint8_t* output_RGB_data_back;
    cv::Mat output_RGB_left;
    uint8_t* output_RGB_data_left;
    cv::Mat output_RGB_right;
    uint8_t* output_RGB_data_right;
    cv::Mat output_RGB_top;
    uint8_t* output_RGB_data_top;
    cv::Mat output_RGB_bottom;
    uint8_t* output_RGB_data_bottom;

    cl_mem _cl_mem_output_RGB_data_front;
    cl_mem _cl_mem_output_RGB_data_back;
    cl_mem _cl_mem_output_RGB_data_left;
    cl_mem _cl_mem_output_RGB_data_right;
    cl_mem _cl_mem_output_RGB_data_top;
    cl_mem _cl_mem_output_RGB_data_bottom;

    void porject_flow_equirect_to_cubemap()
    {
        size_t global_work_size[1] = {static_cast<size_t>(256 * 256)};

        // top
        clSetKernelArg(k_e2p_flow, 0, sizeof(cl_mem), (void *)&_cl_mem_flow_input);
        clSetKernelArg(k_e2p_flow, 1, sizeof(cl_mem), (void *)&_cl_mem_output_flow_top);
        clSetKernelArg(k_e2p_flow, 2, sizeof(cl_mem), (void *)&_cl_mem_output_flow_left);
        clSetKernelArg(k_e2p_flow, 3, sizeof(cl_mem), (void *)&_cl_mem_output_flow_front);
        clSetKernelArg(k_e2p_flow, 4, sizeof(cl_mem), (void *)&_cl_mem_output_flow_right);
        clSetKernelArg(k_e2p_flow, 5, sizeof(cl_mem), (void *)&_cl_mem_output_flow_back);
        clSetKernelArg(k_e2p_flow, 6, sizeof(cl_mem), (void *)&_cl_mem_output_flow_bottom);

        _enentPoint = (cl_event *)malloc(1 * sizeof(cl_event));

        clEnqueueNDRangeKernel(OpenclManager::getInstance()->getCommandQueue(), k_e2p_flow, 1, NULL, global_work_size, NULL, 0, NULL, _enentPoint);

        clEnqueueWaitForEvents(OpenclManager::getInstance()->getCommandQueue(), 1, _enentPoint);

        clWaitForEvents(1, _enentPoint);
        clReleaseEvent(_enentPoint[0]);
        free(_enentPoint);

        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_flow_top, CL_TRUE, 0, 256 * 256 * sizeof(FlowData), output_flow_top, 0, NULL, NULL);
        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_flow_left, CL_TRUE, 0, 256 * 256 * sizeof(FlowData), output_flow_left, 0, NULL, NULL);
        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_flow_front, CL_TRUE, 0, 256 * 256 * sizeof(FlowData), output_flow_front, 0, NULL, NULL);
        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_flow_right, CL_TRUE, 0, 256 * 256 * sizeof(FlowData), output_flow_right, 0, NULL, NULL);
        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_flow_back, CL_TRUE, 0, 256 * 256 * sizeof(FlowData), output_flow_back, 0, NULL, NULL);
        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_output_flow_bottom, CL_TRUE, 0, 256 * 256 * sizeof(FlowData), output_flow_bottom, 0, NULL, NULL);
    }

    FlowData *output_flow_front;
    FlowData *output_flow_back;
    FlowData *output_flow_left;
    FlowData *output_flow_right;
    FlowData *output_flow_top;
    FlowData *output_flow_bottom;

    cl_mem _cl_mem_output_flow_front;
    cl_mem _cl_mem_output_flow_back;
    cl_mem _cl_mem_output_flow_left;
    cl_mem _cl_mem_output_flow_right;
    cl_mem _cl_mem_output_flow_top;
    cl_mem _cl_mem_output_flow_bottom;
};
