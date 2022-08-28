#include <pybind11/pybind11.h>

#include "equirect2cylinder.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/euler_angles.hpp"

#define __FLOW_FILE_TAG__ "PIEH"
const static float __PI__ = 3.1415926535897932;
const static float __PI_2__ = 1.57079632679489661923;
const static float __PI_4__ = 0.78539816339744830962;
const static float __RADIUS__ = 1.0;

glm::mat3 FRONT_ROTATION = glm::mat3(glm::eulerAngleXYZ(0.0f, 0.0f, 0.0f));
glm::mat3 RIGHT_ROTATION = glm::mat3(glm::eulerAngleXYZ(0.0f, glm::radians(90.0f), 0.0f));
glm::mat3 LEFT_ROTATION = glm::mat3(glm::eulerAngleXYZ(0.0f, glm::radians(-90.0f), 0.0f));
glm::mat3 BACK_ROTATION = glm::mat3(glm::eulerAngleXYZ(0.0f, glm::radians(180.0f), 0.0f));
glm::mat3 TOP_ROTATION = glm::mat3(glm::eulerAngleXYZ(glm::radians(90.0f), 0.0f, 0.0f));
glm::mat3 BOTTOM_ROTATION = glm::mat3(glm::eulerAngleXYZ(glm::radians(-90.0f), 0.0f, 0.0f));

#define WRITE(x, y) cv::imwrite(x, y)

enum CameraDirection
{
    TOP = 0,
    LEFT,
    FRONT,
    RIGHT,
    BACK,
    DOWN,
    MAX_DIRECTION_NUM // 6
};

void convertToRadian(const CameraDirection &direction, const float &x, const float &y, float &theta, float &phi)
{
    glm::vec3 xyz = glm::vec3(x, y, 1);
    glm::vec3 ray3d;
    glm::mat3 RK;

    float xp = 0;
    float yp = 0;
    float zp = 0;

    switch (direction)
    {
    case (CameraDirection::TOP):
    {
        RK = TOP_ROTATION;
        break;
    }
    case (CameraDirection::LEFT):
    {
        RK = LEFT_ROTATION;
        break;
    }
    case (CameraDirection::FRONT):
    {
        RK = FRONT_ROTATION;
        break;
    }
    case (CameraDirection::RIGHT):
    {
        RK = RIGHT_ROTATION;
        break;
    }
    case (CameraDirection::BACK):
    {
        RK = BACK_ROTATION;
        break;
    }
    case (CameraDirection::DOWN):
    {
        RK = BOTTOM_ROTATION;
        break;
    }
    default:
    {
    }
    }

    ray3d = RK * xyz;
    xp = ray3d.x;
    yp = ray3d.y;
    zp = ray3d.z;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    if (theta < 0)
        theta += 1024;
    if (theta >= 1024)
        theta-= 1024;
    if (phi < 0)
        phi += 512;
    if (phi >= 512)
        phi -= 512;
}

void project_RGB_equirect_to_cubemap(const cv::Mat &input_RGB_equirect, cv::Mat &output_RGB, const CameraDirection &direction)
{
    int height = 256;
    int width = 256;

    // unsigned long start = us_ticker_read();
// #pragma omp parallel for
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {

            float theta;
            float phi;

            convertToRadian(direction, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), theta, phi);

            output_RGB.at<cv::Vec3b>(row, col) = input_RGB_equirect.at<cv::Vec3b>(phi, theta);
        }
    }
    // final_result += us_ticker_read() - start;
}

void porject_flow_equirect_to_cubemap(const FlowData *input_flow_equirect, FlowData *output_flow, const CameraDirection &direction)
{
    int height = 256;
    int width = 256;

    // unsigned long start = us_ticker_read();

// #pragma omp parallel for
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            float theta; // col
            float phi;   // row

            convertToRadian(direction, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), theta, phi);

            float x = input_flow_equirect[int((phi)) * 1024 + int((theta))].x;
            float y = input_flow_equirect[int((phi)) * 1024 + int((theta))].y;
            float tar_col = x + theta;
            float tar_row = y + phi;

            if (tar_row < 0)
            {
                tar_row = 0;
            }
            if (tar_row > 512)
            {
                tar_row = 512;
            }
            if (tar_col < 0)
            {
                tar_col += 1024;
            }
            if (tar_col > 1024)
            {
                tar_col -= 1024;
            }

            float tar_theta = ((tar_col / 1024) * 2 - 1) * __PI__;
            float tar_phi = ((tar_row / 512) * 2 - 1) * (__PI__ / 2);

            float tar_x = cos(tar_phi) * sin(tar_theta);
            float tar_y = sin(tar_phi);
            float tar_z = cos(tar_phi) * cos(tar_theta);

            float outx = 0;
            float outy = 0;
            switch (direction)
            {
            case (CameraDirection::TOP):
            {
                glm::vec3 ray = glm::vec3(tar_x, tar_y, -tar_z);
                ray = TOP_ROTATION * ray;
                tar_x = ray.x;
                tar_y = ray.y;
                tar_z = -ray.z;

                float ratio = 128.0 / tar_z;
                float rad = 0;

                outx = ratio * tar_x;
                outy = ratio * tar_y;

                float temp = outx;
                outx = outx * cos(rad) - outy * sin(rad);
                outy = temp * sin(rad) + outy * cos(rad);

                outx = outx - (col - 128);
                outy = outy - (row - 128);

                break;
            }
            case (CameraDirection::LEFT):
            {
                glm::vec3 ray = glm::vec3(tar_x, tar_y, -tar_z);
                ray = LEFT_ROTATION * ray;
                tar_x = ray.x;
                tar_y = ray.y;
                tar_z = -ray.z;

                float ratio = 128.0 / tar_z;
                float rad = 0;

                outx = ratio * tar_x;
                outy = ratio * tar_y;

                float temp = outx;
                outx = outx * cos(rad) - outy * sin(rad);
                outy = temp * sin(rad) + outy * cos(rad);

                outx = outx - (col - 128);
                outy = outy - (row - 128);

                break;
            }
            case (CameraDirection::FRONT):
            {
                float ratio = 128.0 / tar_z;
                float rad = 0;

                outx = ratio * tar_x;
                outy = ratio * tar_y;

                float temp = outx;
                outx = outx * cos(rad) - outy * sin(rad);
                outy = temp * sin(rad) + outy * cos(rad);

                outx = outx - (col - 128);
                outy = outy - (row - 128);

                break;
            }
            case (CameraDirection::RIGHT):
            {
                glm::vec3 ray = glm::vec3(tar_x, tar_y, -tar_z);
                ray = RIGHT_ROTATION * ray;
                tar_x = ray.x;
                tar_y = ray.y;
                tar_z = -ray.z;

                float ratio = 128.0 / tar_z;
                float rad = 0;

                outx = ratio * tar_x;
                outy = ratio * tar_y;

                float temp = outx;
                outx = outx * cos(rad) - outy * sin(rad);
                outy = temp * sin(rad) + outy * cos(rad);

                outx = outx - (col - 128);
                outy = outy - (row - 128);
                break;
            }
            case (CameraDirection::BACK):
            {
                glm::vec3 ray = glm::vec3(tar_x, tar_y, -tar_z);
                ray = BACK_ROTATION * ray;
                tar_x = ray.x;
                tar_y = ray.y;
                tar_z = -ray.z;

                float ratio = 128.0 / tar_z;
                float rad = 0;

                outx = ratio * tar_x;
                outy = ratio * tar_y;

                float temp = outx;
                outx = outx * cos(rad) - outy * sin(rad);
                outy = temp * sin(rad) + outy * cos(rad);

                outx = outx - (col - 128);
                outy = outy - (row - 128);
                break;
            }
            case (CameraDirection::DOWN):
            {
                glm::vec3 ray = glm::vec3(tar_x, tar_y, -tar_z);
                ray = BOTTOM_ROTATION * ray;
                tar_x = ray.x;
                tar_y = ray.y;
                tar_z = -ray.z;

                float ratio = 128.0 / tar_z;
                float rad = 0;

                outx = ratio * tar_x;
                outy = ratio * tar_y;

                float temp = outx;
                outx = outx * cos(rad) - outy * sin(rad);
                outy = temp * sin(rad) + outy * cos(rad);

                outx = outx - (col - 128);
                outy = outy - (row - 128);
                break;
            }
            default:
            {
            }
            }
            output_flow[row * 256 + col].x = outx;
            output_flow[row * 256 + col].y = outy;
        }
    }

    // final_result += us_ticker_read() - start;

}

void RGB_cubemap_2_cubepadding(cv::Mat front, cv::Mat back, cv::Mat left, cv::Mat right, cv::Mat top, cv::Mat bottom, cv::Mat &output)
{
    int size = front.rows;

    // block 2-2 front
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = front.at<cv::Vec3b>(i - 2 * size, j - 2 * size);
        }
    }

    // block 2-1 left
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 1 * size; j < 2 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = left.at<cv::Vec3b>(i - 2 * size, j - 1 * size);
        }
    }

    // block 2-1 right
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = right.at<cv::Vec3b>(i - 2 * size, j - 3 * size);
        }
    }

    // block 2-0 back
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            output.at<cv::Vec3b>(i, j) = back.at<cv::Vec3b>(i - 2 * size, j);
        }
    }

    // block 1-2 top
// #pragma omp parallel for
    for (int i = 1 * size; i < 2 * size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = top.at<cv::Vec3b>(i - 1 * size, j - 2 * size);
        }
    }

    // block 3-2 bottom
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = bottom.at<cv::Vec3b>(i - 3 * size, j - 2 * size);
        }
    }

    // block 1-0
// #pragma omp parallel for
    for (int i = 1 * size; i < 2 * size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            output.at<cv::Vec3b>(i, j) = top.at<cv::Vec3b>(size - (i - size) - 1, size - j - 1);
        }
    }

    // block 3-0
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            output.at<cv::Vec3b>(i, j) = bottom.at<cv::Vec3b>(size - (i - 3 * size) - 1, size - j - 1);
        }
    }

    // block 0-2
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = back.at<cv::Vec3b>(size - (i)-1, size - (j - 2 * size) - 1);
        }
    }

    // block 0-1
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = size; j < 2 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = left.at<cv::Vec3b>(size - (i)-1, size - (j - size) - 1);
        }
    }

    // block 0-3
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = right.at<cv::Vec3b>(size - (i)-1, size - (j - 3 * size) - 1);
        }
    }

    //=====================
    // block 1-1
// #pragma omp parallel for
    for (int i = size; i < 2 * size; i++)
    {
        for (int j = size; j < 2 * size; j++)
        {
            if (j > i)
            {
                output.at<cv::Vec3b>(i, j) = left.at<cv::Vec3b>(size - (j - size) - 1, i - size);
            }
            else
            {
                output.at<cv::Vec3b>(i, j) = top.at<cv::Vec3b>(j - size, size - (i - size) - 1);
            }
        }
    }

    // block 3-3
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            if (j > i)
            {
                output.at<cv::Vec3b>(i, j) = bottom.at<cv::Vec3b>(j - 3 * size, size - (i - 3 * size) - 1);
            }
            else
            {
                output.at<cv::Vec3b>(i, j) = right.at<cv::Vec3b>(size - (j - 3 * size) - 1, i - 3 * size);
            }
        }
    }

    // block 1-3
// #pragma omp parallel for
    for (int i = size; i < 2 * size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            if ((j - 3 * size) > (size - (i - size) - 1))
            {
                output.at<cv::Vec3b>(i, j) = top.at<cv::Vec3b>(size - (j - 3 * size) - 1, i - size);
            }
            else
            {
                output.at<cv::Vec3b>(i, j) = right.at<cv::Vec3b>(j - 3 * size, size - (i - size) - 1);
            }
        }
    }

    // block 3-1
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = size; j < 2 * size; j++)
        {
            if ((j - size) > (size - (i - 3 * size) - 1))
            {
                output.at<cv::Vec3b>(i, j) = left.at<cv::Vec3b>(j - size, size - (i - 3 * size) - 1);
            }
            else
            {
                output.at<cv::Vec3b>(i, j) = bottom.at<cv::Vec3b>(size - (j - size) - 1, i - 3 * size);
            }
        }
    }

    // block 0-0
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (j > i)
            {
                if (j > size - i - 1)
                {
                    output.at<cv::Vec3b>(i, j) = left.at<cv::Vec3b>(size - i - 1, j);
                }
                else
                {
                    output.at<cv::Vec3b>(i, j) = bottom.at<cv::Vec3b>(i, size - j - 1);
                }
            }
            else
            {
                if (j > size - i - 1)
                {
                    output.at<cv::Vec3b>(i, j) = top.at<cv::Vec3b>(i, size - j - 1);
                }
                else
                {
                    output.at<cv::Vec3b>(i, j) = right.at<cv::Vec3b>(size - i - 1, j);
                }
            }
        }
    }
}

void flow_cubemap_2_cubepadding(FlowData *front_flow, FlowData *back_flow, FlowData *left_flow, FlowData *right_flow, FlowData *top_flow, FlowData *bottom_flow, FlowData *output_optical_flow)
{
    int size = 256;

    // block 2-2 front
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output_optical_flow[i * 4 * 256 + j] = front_flow[(i - 2 * size) * 256 + j - 2 * size];
        }
    }

    // block 2-1 left
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 1 * size; j < 2 * size; j++)
        {
            output_optical_flow[i * 4 * 256 + j] = left_flow[(i - 2 * size) * 256 + j - 1 * size];
        }
    }

    // block 2-1 right
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            output_optical_flow[i * 4 * 256 + j] = right_flow[(i - 2 * size) * 256 + j - 3 * size];
        }
    }

    // block 2-0 back
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            output_optical_flow[i * 4 * 256 + j] = back_flow[(i - 2 * size) * 256 + j];
        }
    }

    // block 1-2 top
// #pragma omp parallel for
    for (int i = 1 * size; i < 2 * size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output_optical_flow[i * 4 * 256 + j] = top_flow[(i - 1 * size) * 256 + j - 2 * size];
        }
    }

    // block 3-2 bottom
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output_optical_flow[i * 4 * 256 + j] = bottom_flow[(i - 3 * size) * 256 + j - 2 * size];
        }
    }

    // block 1-0
// #pragma omp parallel for
    for (int i = 1 * size; i < 2 * size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            output_optical_flow[i * 4 * 256 + j] = top_flow[(size - (i - size) - 1) * 256 + size - j - 1];
        }
    }

    // block 3-0
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            output_optical_flow[i * 4 * 256 + j] = bottom_flow[(size - (i - 3 * size) - 1) * 256 + size - j - 1];
        }
    }

    // block 0-2
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output_optical_flow[i * 4 * 256 + j] = back_flow[(size - (i)-1) * 256 + size - (j - 2 * size) - 1];
        }
    }

    // block 0-1
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = size; j < 2 * size; j++)
        {
            output_optical_flow[i * 4 * 256 + j] = left_flow[(size - (i)-1) * 256 + size - (j - size) - 1];
        }
    }

    // block 0-3
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            output_optical_flow[i * 4 * 256 + j] = right_flow[(size - (i)-1) * 256 + size - (j - 3 * size) - 1];
        }
    }

    //=====================
    // block 1-1
// #pragma omp parallel for
    for (int i = size; i < 2 * size; i++)
    {
        for (int j = size; j < 2 * size; j++)
        {
            if (j > i)
            {
                output_optical_flow[i * 4 * 256 + j] = left_flow[(size - (j - size) - 1) * 256 + i - size];
            }
            else
            {
                output_optical_flow[i * 4 * 256 + j] = top_flow[(j - size) * 256 + size - (i - size) - 1];
            }
        }
    }

    // block 3-3
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            if (j > i)
            {
                output_optical_flow[i * 4 * 256 + j] = bottom_flow[(j - 3 * size) * 256 + size - (i - 3 * size) - 1];
            }
            else
            {
                output_optical_flow[i * 4 * 256 + j] = right_flow[(size - (j - 3 * size) - 1) * 256 + i - 3 * size];
            }
        }
    }

    // block 1-3
// #pragma omp parallel for
    for (int i = size; i < 2 * size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            if ((j - 3 * size) > (size - (i - size) - 1))
            {
                output_optical_flow[i * 4 * 256 + j] = top_flow[(size - (j - 3 * size) - 1) * 256 + i - size];
            }
            else
            {
                output_optical_flow[i * 4 * 256 + j] = right_flow[(j - 3 * size) * 256 + size - (i - size) - 1];
            }
        }
    }

    // block 3-1
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = size; j < 2 * size; j++)
        {
            if ((j - size) > (size - (i - 3 * size) - 1))
            {
                output_optical_flow[i * 4 * 256 + j] = left_flow[(j - size) * 256 + size - (i - 3 * size) - 1];
            }
            else
            {
                output_optical_flow[i * 4 * 256 + j] = bottom_flow[(size - (j - size) - 1) * 256 + i - 3 * size];
            }
        }
    }

    // block 0-0
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (j > i)
            {
                if (j > size - i - 1)
                {
                    output_optical_flow[i * 4 * 256 + j] = left_flow[(size - i - 1) * 256 + j];
                }
                else
                {
                    output_optical_flow[i * 4 * 256 + j] = bottom_flow[(i)*256 + size - j - 1];
                }
            }
            else
            {
                if (j > size - i - 1)
                {
                    output_optical_flow[i * 4 * 256 + j] = top_flow[(i)*256 + size - j - 1];
                }
                else
                {
                    output_optical_flow[i * 4 * 256 + j] = right_flow[(size - i - 1) * 256 + j];
                }
            }
        }
    }
}

void cubemap_2_cubepadding(
    cv::Mat front, cv::Mat back, cv::Mat left, cv::Mat right, cv::Mat top, cv::Mat bottom,
    FlowData *front_flow, FlowData *back_flow, FlowData *left_flow, FlowData *right_flow, FlowData *top_flow, FlowData *bottom_flow,
    cv::Mat &output, FlowData *output_optical_flow)
{
    int size = front.rows;

    // block 2-2 front
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = front.at<cv::Vec3b>(i - 2 * size, j - 2 * size);
            output_optical_flow[i * 4 * 256 + j] = front_flow[(i - 2 * size) * 256 + j - 2 * size];
        }
    }

    // block 2-1 left
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 1 * size; j < 2 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = left.at<cv::Vec3b>(i - 2 * size, j - 1 * size);
            output_optical_flow[i * 4 * 256 + j] = left_flow[(i - 2 * size) * 256 + j - 1 * size];
        }
    }

    // block 2-1 right
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = right.at<cv::Vec3b>(i - 2 * size, j - 3 * size);
            output_optical_flow[i * 4 * 256 + j] = right_flow[(i - 2 * size) * 256 + j - 3 * size];
        }
    }

    // block 2-0 back
// #pragma omp parallel for
    for (int i = 2 * size; i < 3 * size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            output.at<cv::Vec3b>(i, j) = back.at<cv::Vec3b>(i - 2 * size, j);
            output_optical_flow[i * 4 * 256 + j] = back_flow[(i - 2 * size) * 256 + j];
        }
    }

    // block 1-2 top
// #pragma omp parallel for
    for (int i = 1 * size; i < 2 * size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = top.at<cv::Vec3b>(i - 1 * size, j - 2 * size);
            output_optical_flow[i * 4 * 256 + j] = top_flow[(i - 1 * size) * 256 + j - 2 * size];
        }
    }

    // block 3-2 bottom
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = bottom.at<cv::Vec3b>(i - 3 * size, j - 2 * size);
            output_optical_flow[i * 4 * 256 + j] = bottom_flow[(i - 3 * size) * 256 + j - 2 * size];
        }
    }

    // block 1-0
// #pragma omp parallel for
    for (int i = 1 * size; i < 2 * size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            output.at<cv::Vec3b>(i, j) = top.at<cv::Vec3b>(size - (i - size) - 1, size - j - 1);
            output_optical_flow[i * 4 * 256 + j] = top_flow[(size - (i - size) - 1) * 256 + size - j - 1];
        }
    }

    // block 3-0
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            output.at<cv::Vec3b>(i, j) = bottom.at<cv::Vec3b>(size - (i - 3 * size) - 1, size - j - 1);
            output_optical_flow[i * 4 * 256 + j] = bottom_flow[(size - (i - 3 * size) - 1) * 256 + size - j - 1];
        }
    }

    // block 0-2
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 2 * size; j < 3 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = back.at<cv::Vec3b>(size - (i)-1, size - (j - 2 * size) - 1);
            output_optical_flow[i * 4 * 256 + j] = back_flow[(size - (i)-1) * 256 + size - (j - 2 * size) - 1];
        }
    }

    // block 0-1
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = size; j < 2 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = left.at<cv::Vec3b>(size - (i)-1, size - (j - size) - 1);
            output_optical_flow[i * 4 * 256 + j] = left_flow[(size - (i)-1) * 256 + size - (j - size) - 1];
        }
    }

    // block 0-3
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            output.at<cv::Vec3b>(i, j) = right.at<cv::Vec3b>(size - (i)-1, size - (j - 3 * size) - 1);
            output_optical_flow[i * 4 * 256 + j] = right_flow[(size - (i)-1) * 256 + size - (j - 3 * size) - 1];
        }
    }

    //=====================
    // block 1-1
// #pragma omp parallel for
    for (int i = size; i < 2 * size; i++)
    {
        for (int j = size; j < 2 * size; j++)
        {
            if (j > i)
            {
                output.at<cv::Vec3b>(i, j) = left.at<cv::Vec3b>(size - (j - size) - 1, i - size);
                output_optical_flow[i * 4 * 256 + j] = left_flow[(size - (j - size) - 1) * 256 + i - size];
            }
            else
            {
                output.at<cv::Vec3b>(i, j) = top.at<cv::Vec3b>(j - size, size - (i - size) - 1);
                output_optical_flow[i * 4 * 256 + j] = top_flow[(j - size) * 256 + size - (i - size) - 1];
            }
        }
    }

    // block 3-3
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            if (j > i)
            {
                output.at<cv::Vec3b>(i, j) = bottom.at<cv::Vec3b>(j - 3 * size, size - (i - 3 * size) - 1);
                output_optical_flow[i * 4 * 256 + j] = bottom_flow[(j - 3 * size) * 256 + size - (i - 3 * size) - 1];
            }
            else
            {
                output.at<cv::Vec3b>(i, j) = right.at<cv::Vec3b>(size - (j - 3 * size) - 1, i - 3 * size);
                output_optical_flow[i * 4 * 256 + j] = right_flow[(size - (j - 3 * size) - 1) * 256 + i - 3 * size];
            }
        }
    }

    // block 1-3
// #pragma omp parallel for
    for (int i = size; i < 2 * size; i++)
    {
        for (int j = 3 * size; j < 4 * size; j++)
        {
            if ((j - 3 * size) > (size - (i - size) - 1))
            {
                output.at<cv::Vec3b>(i, j) = top.at<cv::Vec3b>(size - (j - 3 * size) - 1, i - size);
                output_optical_flow[i * 4 * 256 + j] = top_flow[(size - (j - 3 * size) - 1) * 256 + i - size];
            }
            else
            {
                output.at<cv::Vec3b>(i, j) = right.at<cv::Vec3b>(j - 3 * size, size - (i - size) - 1);
                output_optical_flow[i * 4 * 256 + j] = right_flow[(j - 3 * size) * 256 + size - (i - size) - 1];
            }
        }
    }

    // block 3-1
// #pragma omp parallel for
    for (int i = 3 * size; i < 4 * size; i++)
    {
        for (int j = size; j < 2 * size; j++)
        {
            if ((j - size) > (size - (i - 3 * size) - 1))
            {
                output.at<cv::Vec3b>(i, j) = left.at<cv::Vec3b>(j - size, size - (i - 3 * size) - 1);
                output_optical_flow[i * 4 * 256 + j] = left_flow[(j - size) * 256 + size - (i - 3 * size) - 1];
            }
            else
            {
                output.at<cv::Vec3b>(i, j) = bottom.at<cv::Vec3b>(size - (j - size) - 1, i - 3 * size);
                output_optical_flow[i * 4 * 256 + j] = bottom_flow[(size - (j - size) - 1) * 256 + i - 3 * size];
            }
        }
    }

    // block 0-0
// #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (j > i)
            {
                if (j > size - i - 1)
                {
                    output.at<cv::Vec3b>(i, j) = left.at<cv::Vec3b>(size - i - 1, j);
                    output_optical_flow[i * 4 * 256 + j] = left_flow[(size - i - 1) * 256 + j];
                }
                else
                {
                    output.at<cv::Vec3b>(i, j) = bottom.at<cv::Vec3b>(i, size - j - 1);
                    output_optical_flow[i * 4 * 256 + j] = bottom_flow[(i)*256 + size - j - 1];
                }
            }
            else
            {
                if (j > size - i - 1)
                {
                    output.at<cv::Vec3b>(i, j) = top.at<cv::Vec3b>(i, size - j - 1);
                    output_optical_flow[i * 4 * 256 + j] = top_flow[(i)*256 + size - j - 1];
                }
                else
                {
                    output.at<cv::Vec3b>(i, j) = right.at<cv::Vec3b>(size - i - 1, j);
                    output_optical_flow[i * 4 * 256 + j] = right_flow[(size - i - 1) * 256 + j];
                }
            }
        }
    }
}



FlowData* synthesizeCyli(FlowData* flow) {
    FlowData* ret = (FlowData*)malloc(3*_proj->getOutputImageHeight() * _proj->getOutputImageWidth() * sizeof(FlowData));

    const CubemapsFaceCoordThetaPhi* coord_theta_phi = NULL;

    uint32_t _image_width  = _proj->getImageWidth();
    uint32_t _image_height = _proj->getImageHeight();
    uint32_t _output_width = _proj->getOutputImageWidth();
    uint32_t _output_height = _proj->getOutputImageHeight();

    // unsigned long start = us_ticker_read();

    // front
// #pragma omp parallel for
    for (uint32_t i = 0; i < _proj->getOutputImageWidth(); ++i) { // col
        for (uint32_t j = 0; j < _proj->getOutputImageHeight(); ++j) { // row
            // CGRA_LOG("=======");
            float middle = (float(j) + 0.5) - (_output_height / 2); // (-h/2, h/2)
            float original_phi = 2 * (atan(exp(float(middle / (_output_width/2))))) - __PI_2__;
            original_phi = (original_phi / __PI_2__ + 1.0) / 2.0 * _image_height;

            float original_theta = i + 0.5f;

            // get flow address position
            coord_theta_phi = _proj->getCoord_theta_phi_front(i, j);
            float col = ((coord_theta_phi->theta / __PI__ + 1.0) / 2.0) * _proj->getImageWidth();
            float row = ((-coord_theta_phi->phi / __PI_2__ + 1.0) / 2.0) * _proj->getImageHeight();

            // bi-interp
            float bi_x;
            float bi_y;
            if (row > 0.5 && row < (511.0 - 0.5) && col > 0.5 && col < (1023.0 - 0.5))
            {
                float x = col - 0.5 - uint32_t(col);
                float y = row - 0.5 - uint32_t(row);
                x += (x < 0.0 ? 1.0 : 0.0);
                y += (y < 0.0 ? 1.0 : 0.0);
                uint32_t xp = uint32_t(col);
                uint32_t yp = uint32_t(row);
                float bi_x_a = x * flow[yp * _image_width + xp].x + (1.0 - x) * flow[yp * _image_width + xp + 1].x;
                float bi_y_a = x * flow[yp * _image_width + xp].y + (1.0 - x) * flow[yp * _image_width + xp + 1].y;
                float bi_x_b = x * flow[(yp + 1) * _image_width + xp].x + (1.0 - x) * flow[(yp + 1) * _image_width + xp + 1].x;
                float bi_y_b = x * flow[(yp + 1) * _image_width + xp].y + (1.0 - x) * flow[(yp + 1) * _image_width + xp + 1].y;
                bi_x = y * bi_x_a + (1.0 - y) * bi_x_b;
                bi_y = y * bi_y_a + (1.0 - y) * bi_y_b;
            }
            else
            {
                bi_x = flow[uint32_t(row) * _image_width + uint32_t(col)].x;
                bi_y = flow[uint32_t(row) * _image_width + uint32_t(col)].y;
            }

            // pick up value
            float target_theta = bi_x + original_theta;
            float target_phi = bi_y + original_phi;

            // CGRA_LOG("%u %u -> target: %f %f", col, row, target_theta, target_phi);

            // back to (-pi, pi) and (-pi/2, pi/2)
            target_theta = (target_theta / _image_width * 2.0 - 1.0) * __PI__;
            target_phi = (target_phi / _image_height * 2.0 - 1.0) * __PI_2__;

            // to xyz (left hand)
            float x = cos(target_phi) * sin(target_theta);
            float y = sin(target_phi);
            float z = cos(target_phi) * cos(target_theta);

            // to right hand
            z = -z;

            // CGRA_LOG("-> xyz: %f %f %f", x, y, z);

            // apply rotation
            // todo

            // to left hand
            z = -z;

            // to theta and phi
            float new_phi = atan2(y, sqrt(x * x + z * z));
            float new_theta = atan2(x, z);
            // CGRA_LOG("-> new theta phi: %f %f", new_theta, new_phi);

            // to Cyli
            float target_row = ((_output_width/2.0) * log(tan(new_phi) + (1.0 / cos(new_phi)))) + (_output_height / 2);
            // CGRA_LOG("new_phi:%f target_row:%f", new_phi, target_row);


            float target_col = new_theta / __PI__ * 512.0 + 512.0;

            // CGRA_LOG("-> pos: %f %f ", target_col, target_row);

            // CGRA_LOG("=======");
            ret[j * _image_width + i].x = target_col - i;
            ret[j * _image_width + i].y = target_row - j;

            if (ret[j * _image_width + i].x < -512) ret[j * _image_width + i].x += 1024;
            if (ret[j * _image_width + i].x > 512) ret[j * _image_width + i].x -= 1024;
        }
    }

    // top
// #pragma omp parallel for
    for (uint32_t i = 0; i < _proj->getOutputImageWidth(); ++i) { // col
        for (uint32_t temp_j = _proj->getOutputImageHeight(); temp_j < 2 * _proj->getOutputImageHeight(); ++temp_j) { // row
            uint32_t j = temp_j - _proj->getOutputImageHeight();
            // CGRA_LOG("=======");
            float middle = (float(j) + 0.5) - (_output_height / 2); // (-h/2, h/2)
            float original_phi = 2 * (atan(exp(float(middle / (_output_width/2))))) - __PI_2__;
            // original_phi = (original_phi / __PI_2__ + 1.0) / 2.0 * _image_height;
            float original_theta = ((float(i) + 0.5f) / _image_width * 2.0 - 1.0) * __PI__;
            float original_x = cos(original_phi) * sin(original_theta);
            float original_y = sin(original_phi);
            float original_z = cos(original_phi) * cos(original_theta);
            glm::vec4 original_vec(original_x,original_y,original_z, 1.0f);
            original_vec = glm::normalize(glm::eulerAngleXYZ(glm::radians(90.0f), 0.0f, 0.0f) * original_vec);
            original_vec = original_vec / original_vec.w;
            original_x = original_vec.x;
            original_y = original_vec.y;
            original_z = original_vec.z;
            original_phi = atan2(original_y, sqrt(original_x * original_x + original_z * original_z));
            original_theta = atan2(original_x, original_z);
            original_phi = ((original_phi / __PI_2__ + 1.0) / 2.0) * _proj->getImageHeight();
            original_theta = ((original_theta / __PI__ + 1.0) / 2.0) * _proj->getImageWidth();



            // get flow address position
            coord_theta_phi = _proj->getCoord_theta_phi_top(i, j);
            float col = ((coord_theta_phi->theta / __PI__ + 1.0) / 2.0) * _proj->getImageWidth();
            float row = ((-coord_theta_phi->phi / __PI_2__ + 1.0) / 2.0) * _proj->getImageHeight();

            // bi-interp
            float bi_x;
            float bi_y;
            if (row > 0.5 && row < (511.0 - 0.5) && col > 0.5 && col < (1023.0 - 0.5))
            {
                float x = col - 0.5 - uint32_t(col);
                float y = row - 0.5 - uint32_t(row);
                x += (x < 0.0 ? 1.0 : 0.0);
                y += (y < 0.0 ? 1.0 : 0.0);
                uint32_t xp = uint32_t(col);
                uint32_t yp = uint32_t(row);
                float bi_x_a = x * flow[yp * _image_width + xp].x + (1.0 - x) * flow[yp * _image_width + xp + 1].x;
                float bi_y_a = x * flow[yp * _image_width + xp].y + (1.0 - x) * flow[yp * _image_width + xp + 1].y;
                float bi_x_b = x * flow[(yp + 1) * _image_width + xp].x + (1.0 - x) * flow[(yp + 1) * _image_width + xp + 1].x;
                float bi_y_b = x * flow[(yp + 1) * _image_width + xp].y + (1.0 - x) * flow[(yp + 1) * _image_width + xp + 1].y;
                bi_x = y * bi_x_a + (1.0 - y) * bi_x_b;
                bi_y = y * bi_y_a + (1.0 - y) * bi_y_b;
            }
            else
            {
                bi_x = flow[uint32_t(row) * _image_width + uint32_t(col)].x;
                bi_y = flow[uint32_t(row) * _image_width + uint32_t(col)].y;
            }

            // pick up value
            float target_theta = bi_x + original_theta;
            float target_phi = bi_y + original_phi;

            // CGRA_LOG("%u %u -> target: %f %f", col, row, target_theta, target_phi);

            // back to (-pi, pi) and (-pi/2, pi/2)
            target_theta = (target_theta / _image_width * 2.0 - 1.0) * __PI__;
            target_phi = (target_phi / _image_height * 2.0 - 1.0) * __PI_2__;

            // to xyz (left hand)
            float x = cos(target_phi) * sin(target_theta);
            float y = sin(target_phi);
            float z = cos(target_phi) * cos(target_theta);

            // to right hand
            z = -z;

            // CGRA_LOG("-> xyz: %f %f %f", x, y, z);

            // apply rotation
            glm::vec4 vec(x,y,z, 1.0f);
            vec = glm::normalize(glm::eulerAngleXYZ(glm::radians(90.0f), 0.0f, 0.0f) * vec);
            vec = vec / vec.w;
            x = vec.x;
            y = vec.y;
            z = vec.z;

            // to left hand
            z = -z;

            // to theta and phi
            float new_phi = atan2(y, sqrt(x * x + z * z));
            float new_theta = atan2(x, z);
            // CGRA_LOG("-> new theta phi: %f %f", new_theta, new_phi);

            // to Cyli
            float target_row = ((_output_width/2.0) * log(tan(new_phi) + (1.0 / cos(new_phi)))) + (_output_height / 2);
            // CGRA_LOG("new_phi:%f target_row:%f", new_phi, target_row);


            float target_col = new_theta / __PI__ * 512.0 + 512.0;

            // CGRA_LOG("-> pos: %f %f ", target_col, target_row);

            // CGRA_LOG("=======");
            ret[temp_j * _image_width + i].x = target_col - i;
            ret[temp_j * _image_width + i].y = target_row - j;

            if (ret[temp_j * _image_width + i].x < -512) ret[temp_j * _image_width + i].x += 1024;
            if (ret[temp_j * _image_width + i].x > 512) ret[temp_j * _image_width + i].x -= 1024;
        }
    }

    // left
// #pragma omp parallel for
    for (uint32_t i = 0; i < _proj->getOutputImageWidth(); ++i) { // col
        for (uint32_t temp_j = 2 * _proj->getOutputImageHeight(); temp_j < 3 * _proj->getOutputImageHeight(); ++temp_j) { // row
            uint32_t j = temp_j - 2 * _proj->getOutputImageHeight();
            // CGRA_LOG("=======");
            float middle = (float(j) + 0.5) - (_output_height / 2); // (-h/2, h/2)
            float original_phi = 2 * (atan(exp(float(middle / (_output_width/2))))) - __PI_2__;
            original_phi = -original_phi;
            // original_phi = (original_phi / __PI_2__ + 1.0) / 2.0 * _image_height;
            float original_theta = ((float(i) + 0.5f) / _image_width * 2.0 - 1.0) * __PI__;
            float original_x = cos(original_phi) * sin(original_theta);
            float original_y = sin(original_phi);
            float original_z = cos(original_phi) * cos(original_theta);
            glm::vec4 original_vec(original_x,original_y,original_z, 1.0f);
            original_vec = glm::normalize(glm::eulerAngleXYZ(0.0f, 0.0f, glm::radians(-90.0f)) * original_vec);
            original_vec = original_vec / original_vec.w;
            original_x = original_vec.x;
            original_y = original_vec.y;
            original_z = original_vec.z;
            original_phi = atan2(original_y, sqrt(original_x * original_x + original_z * original_z));
            original_phi = -original_phi;
            original_theta = atan2(original_x, original_z);
            original_phi = ((original_phi / __PI_2__ + 1.0) / 2.0) * _proj->getImageHeight();
            original_theta = ((original_theta / __PI__ + 1.0) / 2.0) * _proj->getImageWidth();



            // get flow address position
            coord_theta_phi = _proj->getCoord_theta_phi_left(i, j);
            float col = ((coord_theta_phi->theta / __PI__ + 1.0) / 2.0) * _proj->getImageWidth();
            float row = ((-coord_theta_phi->phi / __PI_2__ + 1.0) / 2.0) * _proj->getImageHeight();

            // bi-interp
            float bi_x;
            float bi_y;
            if (row > 0.5 && row < (511.0 - 0.5) && col > 0.5 && col < (1023.0 - 0.5))
            {
                float x = col - 0.5 - uint32_t(col);
                float y = row - 0.5 - uint32_t(row);
                x += (x < 0.0 ? 1.0 : 0.0);
                y += (y < 0.0 ? 1.0 : 0.0);
                uint32_t xp = uint32_t(col);
                uint32_t yp = uint32_t(row);
                float bi_x_a = x * flow[yp * _image_width + xp].x + (1.0 - x) * flow[yp * _image_width + xp + 1].x;
                float bi_y_a = x * flow[yp * _image_width + xp].y + (1.0 - x) * flow[yp * _image_width + xp + 1].y;
                float bi_x_b = x * flow[(yp + 1) * _image_width + xp].x + (1.0 - x) * flow[(yp + 1) * _image_width + xp + 1].x;
                float bi_y_b = x * flow[(yp + 1) * _image_width + xp].y + (1.0 - x) * flow[(yp + 1) * _image_width + xp + 1].y;
                bi_x = y * bi_x_a + (1.0 - y) * bi_x_b;
                bi_y = y * bi_y_a + (1.0 - y) * bi_y_b;
            }
            else
            {
                bi_x = flow[uint32_t(row) * _image_width + uint32_t(col)].x;
                bi_y = flow[uint32_t(row) * _image_width + uint32_t(col)].y;
            }

            // pick up value
            float target_theta = bi_x + original_theta;
            float target_phi = bi_y + original_phi;

            // CGRA_LOG("%u %u -> target: %f %f", col, row, target_theta, target_phi);

            // back to (-pi, pi) and (-pi/2, pi/2)
            target_theta = (target_theta / _image_width * 2.0 - 1.0) * __PI__;
            target_phi = (target_phi / _image_height * 2.0 - 1.0) * __PI_2__;

            // to xyz (left hand)
            float x = cos(target_phi) * sin(target_theta);
            float y = sin(target_phi);
            float z = cos(target_phi) * cos(target_theta);

            // to right hand
            z = -z;

            // CGRA_LOG("-> xyz: %f %f %f", x, y, z);

            // apply rotation
            glm::vec4 vec(x,y,z, 1.0f);
            vec = glm::normalize(glm::eulerAngleXYZ(0.0f, 0.0f, glm::radians(-90.0f)) * vec);
            vec = vec / vec.w;
            x = vec.x;
            y = vec.y;
            z = vec.z;

            // to left hand
            z = -z;

            // to theta and phi
            float new_phi = atan2(y, sqrt(x * x + z * z));
            float new_theta = atan2(x, z);
            // CGRA_LOG("-> new theta phi: %f %f", new_theta, new_phi);

            // to Cyli
            float target_row = ((_output_width/2.0) * log(tan(new_phi) + (1.0 / cos(new_phi)))) + (_output_height / 2);
            // CGRA_LOG("new_phi:%f target_row:%f", new_phi, target_row);


            float target_col = new_theta / __PI__ * 512.0 + 512.0;

            // CGRA_LOG("-> pos: %f %f ", target_col, target_row);

            // CGRA_LOG("=======");
            ret[temp_j * _image_width + i].x = target_col - i;
            ret[temp_j * _image_width + i].y = target_row - j;

            if (ret[temp_j * _image_width + i].x < -512) ret[temp_j * _image_width + i].x += 1024;
            if (ret[temp_j * _image_width + i].x > 512) ret[temp_j * _image_width + i].x -= 1024;
        }
    }

    // final_result += us_ticker_read() - start;

    return ret;
}


cv::Mat generate_cylin_RGB(cv::Mat present)
{

    // cv::Mat present = cv::imread(std::string(str.str()).c_str());
    cv::Mat cyli = cv::Mat(3 * _proj->getOutputImageHeight(), _proj->getOutputImageWidth(), CV_8UC3, cv::Scalar(0, 0, 0));

    const CubemapsFaceCoordThetaPhi *coord_theta_phi = NULL;

    // unsigned long start = us_ticker_read();

// #pragma omp parallel for
    for (uint32_t i = 0; i < _proj->getOutputImageWidth(); ++i)
    {
        for (uint32_t j = 0; j < _proj->getOutputImageHeight(); ++j)
        {
            // coord = _proj->getCoord_front(i, j);
            // cyli.at<cv::Vec3b>(j, i) = bmpCube[coord->face].at<cv::Vec3b>(coord->y, coord->x);

            coord_theta_phi = _proj->getCoord_theta_phi_front(i, j);
            float col = ((coord_theta_phi->theta / __PI__ + 1.0) / 2.0) * _proj->getImageWidth();
            float row = ((-coord_theta_phi->phi / __PI_2__ + 1.0) / 2.0) * _proj->getImageHeight();

            if (row > 0.5 && row < (511.0 - 0.5) && col > 0.5 && col < (1023.0 - 0.5))
            {
                float x = col - 0.5 - uint32_t(col);
                float y = row - 0.5 - uint32_t(row);
                x += (x < 0.0 ? 1.0 : 0.0);
                y += (y < 0.0 ? 1.0 : 0.0);
                uint32_t xp = uint32_t(col);
                uint32_t yp = uint32_t(row);
                cv::Vec3f a = x * present.at<cv::Vec3b>(yp, xp) + (1.0 - x) * present.at<cv::Vec3b>(yp, xp + 1);
                cv::Vec3f b = x * present.at<cv::Vec3b>(yp + 1, xp) + (1.0 - x) * present.at<cv::Vec3b>(yp + 1, xp + 1);
                cv::Vec3b c = y * a + (1.0 - y) * b;

                cyli.at<cv::Vec3b>(j, i) = present.at<cv::Vec3b>(row, col);
            }
            else
            {
                cyli.at<cv::Vec3b>(j, i) = present.at<cv::Vec3b>(row, col);
            }
        }
    }

// #pragma omp parallel for
    for (uint32_t i = 0; i < _proj->getOutputImageWidth(); ++i)
    {
        for (uint32_t j = _proj->getOutputImageHeight(); j < 2 * _proj->getOutputImageHeight(); ++j)
        {
            // coord = _proj->getCoord_top(i, j - _proj->getOutputImageHeight());
            // cyli.at<cv::Vec3b>(j, i) = bmpCube[coord->face].at<cv::Vec3b>(coord->y, coord->x);

            coord_theta_phi = _proj->getCoord_theta_phi_top(i, j - _proj->getOutputImageHeight());
            float col = ((coord_theta_phi->theta / __PI__ + 1.0) / 2.0) * _proj->getImageWidth();
            float row = ((-coord_theta_phi->phi / __PI_2__ + 1.0) / 2.0) * _proj->getImageHeight();

            if (row > 0.5 && row < (511.0 - 0.5) && col > 0.5 && col < (1023.0 - 0.5))
            {
                float x = col - 0.5 - uint32_t(col);
                float y = row - 0.5 - uint32_t(row);
                x += (x < 0.0 ? 1.0 : 0.0);
                y += (y < 0.0 ? 1.0 : 0.0);
                uint32_t xp = uint32_t(col);
                uint32_t yp = uint32_t(row);
                cv::Vec3f a = x * present.at<cv::Vec3b>(yp, xp) + (1.0 - x) * present.at<cv::Vec3b>(yp, xp + 1);
                cv::Vec3f b = x * present.at<cv::Vec3b>(yp + 1, xp) + (1.0 - x) * present.at<cv::Vec3b>(yp + 1, xp + 1);
                cv::Vec3b c = y * a + (1.0 - y) * b;

                cyli.at<cv::Vec3b>(j, i) = present.at<cv::Vec3b>(row, col);
            }
            else
            {
                cyli.at<cv::Vec3b>(j, i) = present.at<cv::Vec3b>(row, col);
            }
        }
    }

// #pragma omp parallel for
    for (uint32_t i = 0; i < _proj->getOutputImageWidth(); ++i)
    {
        for (uint32_t j = 2 * _proj->getOutputImageHeight(); j < 3 * _proj->getOutputImageHeight(); ++j)
        {
            // coord = _proj->getCoord_left(i, j - 2 * _proj->getOutputImageHeight());
            // cyli.at<cv::Vec3b>(j, i) = bmpCube[coord->face].at<cv::Vec3b>(coord->y, coord->x);

            coord_theta_phi = _proj->getCoord_theta_phi_left(i, j - 2 * _proj->getOutputImageHeight());
            float col = ((coord_theta_phi->theta / __PI__ + 1.0) / 2.0) * _proj->getImageWidth();
            float row = ((-coord_theta_phi->phi / __PI_2__ + 1.0) / 2.0) * _proj->getImageHeight();

            if (row > 0.5 && row < (511.0 - 0.5) && col > 0.5 && col < (1023.0 - 0.5))
            {
                float x = col - 0.5 - uint32_t(col);
                float y = row - 0.5 - uint32_t(row);
                x += (x < 0.0 ? 1.0 : 0.0);
                y += (y < 0.0 ? 1.0 : 0.0);
                uint32_t xp = uint32_t(col);
                uint32_t yp = uint32_t(row);
                cv::Vec3f a = x * present.at<cv::Vec3b>(yp, xp) + (1.0 - x) * present.at<cv::Vec3b>(yp, xp + 1);
                cv::Vec3f b = x * present.at<cv::Vec3b>(yp + 1, xp) + (1.0 - x) * present.at<cv::Vec3b>(yp + 1, xp + 1);
                cv::Vec3b c = y * a + (1.0 - y) * b;

                cyli.at<cv::Vec3b>(j, i) = present.at<cv::Vec3b>(row, col);
            }
            else
            {
                cyli.at<cv::Vec3b>(j, i) = present.at<cv::Vec3b>(row, col);
            }
        }
    }

    return cyli;
}



/////////////////////////////////////////////////////

FlowData* flow_internel_cylinder_to_equirect(float* buff) {
    FlowData* flow_data;
    flow_data = (FlowData*)buff;

    uint32_t _image_width  = _proj->getImageWidth(); // equi
    uint32_t _image_height = _proj->getImageHeight(); // equi
    uint32_t _output_width = _proj->getOutputImageWidth(); // cyli
    uint32_t _output_height = _proj->getOutputImageHeight(); // cyli

    // unsigned long start = us_ticker_read();

    // CGRA_LOG("_image_width:%d _image_height:%d _output_width:%d _output_height:%d", _image_width, _image_height, _output_width, _output_height);
    FlowData* output_flow_data = (FlowData*)malloc(_image_width*_image_height*sizeof(FlowData));

    for (int row = 0; row < _image_height; row++) {
        for (int col = 0; col < _image_width; col++) {
            float _row = row;
            float _col = col;
            _row += 0.5;
            _col += 0.5;
            float theta = ((_col / _image_width) * 2.0 - 1.0) * __PI__;
            float phi = ((_row / _image_height) * 2.0 - 1.0) * __PI_2__;
            phi = -phi;

            // left hand system
            float x = cos(phi) * sin(theta);
            float y = sin(phi);
            float z = cos(phi) * cos(theta);

            float abs_x = abs(x);
            float abs_y = abs(y);
            float abs_z = abs(z);
            // left hand system
            // front view is X-Z plane, read y
            if (abs_y < abs_x && abs_y < abs_z) {
                phi = -phi;

                float cyli_row = ((_output_width/2.0) * log(tan(phi) + (1.0 / cos(phi)))) + (_output_height / 2);
                float cyli_col = theta / __PI__ * 512.0 + 512.0;

                FlowData pixel_flow;
                try {
                    pixel_flow = flow_data[int(cyli_row) * _output_width + int(cyli_col)];
                }
                catch (std::exception e) {
                    CGRA_LOGD("ERROR! %f %f", cyli_row, cyli_col);
                    pixel_flow.x = 0;
                    pixel_flow.y = 0;
                }

                float target_row = cyli_row + pixel_flow.y;
                float target_col = cyli_col + pixel_flow.x;
                if (target_col < 0) target_col += 1024;
                if (target_col >= 1024) target_col -= 1024;

                float target_middle = (float(target_row)) - (_output_height / 2);
                float target_phi = 2 * (atan(exp(float(target_middle / (_output_width/2))))) - __PI_2__;
                float target_theta = ((float(target_col)) / _image_width * 2.0 - 1.0) * __PI__;

                target_row = ((target_phi / __PI_2__) + 1.0) * 0.5 * _image_height;
                target_col = target_col;

                output_flow_data[row * _image_width + col].x = target_col - _col;
                output_flow_data[row * _image_width + col].y = target_row - _row;
                if (output_flow_data[row * _image_width + col].x < -512) output_flow_data[row * _image_width + col].x += 1024;
                if (output_flow_data[row * _image_width + col].x > 512) output_flow_data[row * _image_width + col].x -= 1024;
            }
            // top view is X-Y plane, read z
            else if (abs_z < abs_x && abs_z < abs_y) {
                glm::vec4 ray = glm::vec4(x, y, -z, 1.0f); // to right hand
                glm::mat4 rotation = glm::eulerAngleXYZ(glm::radians(-90.0f), 0.0f, 0.0f);
                glm::vec4 out = glm::normalize(rotation * ray);
                out = out / out.w;

                x = out.x;
                y = out.y;
                z = out.z;

                z = -z;

                phi = atan2(y, sqrt(x * x + z * z));
                theta = atan2(x, z);

                phi = -phi;

                float cyli_row = ((_output_width/2.0) * log(tan(phi) + (1.0 / cos(phi)))) + (_output_height / 2);
                float cyli_col = theta / __PI__ * 512.0 + 512.0;
    
                FlowData pixel_flow;
                try {
                    pixel_flow = flow_data[int(cyli_row + 1 * _output_height) * _output_width + int(cyli_col)];
                }
                catch (std::exception e) {
                    CGRA_LOGD("ERROR! %f %f", cyli_row, cyli_col);
                    pixel_flow.x = 0;
                    pixel_flow.y = 0;
                }

                float target_row = cyli_row + pixel_flow.y;
                float target_col = cyli_col + pixel_flow.x;
                if (target_col < 0) target_col += 1024;
                if (target_col >= 1024) target_col -= 1024;

                float target_middle = (float(target_row)) - (_output_height / 2);
                float target_phi = 2 * (atan(exp(float(target_middle / (_output_width/2))))) - __PI_2__;
                target_phi = -target_phi;
                float target_theta = ((float(target_col)) / _image_width * 2.0 - 1.0) * __PI__;

                x = cos(target_phi) * sin(target_theta);
                y = sin(target_phi);
                z = cos(target_phi) * cos(target_theta);
                ray = glm::vec4(x, y, -z, 1.0f); // to right hand
                rotation = glm::eulerAngleXYZ(glm::radians(90.0f), 0.0f, 0.0f);
                out = glm::normalize(rotation * ray);
                out = out / out.w;
                x = out.x;
                y = out.y;
                z = out.z;

                z = -z;

                target_phi = atan2(y, sqrt(x * x + z * z));
                target_phi = -target_phi;
                target_theta = atan2(x, z);

                target_row = ((target_phi / __PI_2__) + 1.0) * 0.5 * _image_height;
                target_col = ((target_theta/ __PI__) + 1.0) * 0.5 * _image_width;

                output_flow_data[row * _image_width + col].x = target_col - _col;
                output_flow_data[row * _image_width + col].y = target_row - _row;
                if (output_flow_data[row * _image_width + col].x < -512) output_flow_data[row * _image_width + col].x += 1024;
                if (output_flow_data[row * _image_width + col].x > 512) output_flow_data[row * _image_width + col].x -= 1024;
            }
            // left view
            else if (abs_x < abs_y && abs_x < abs_z) {
                glm::vec4 ray = glm::vec4(x, y, -z, 1.0f); // to right hand
                glm::mat4 rotation = glm::eulerAngleXYZ(0.0f, 0.0f, glm::radians(90.0f));
                glm::vec4 out = glm::normalize(rotation * ray);
                out = out / out.w;

                x = out.x;
                y = out.y;
                z = out.z;

                z = -z;

                phi = atan2(y, sqrt(x * x + z * z));
                theta = atan2(x, z);

                phi = -phi;

                float cyli_row = ((_output_width/2.0) * log(tan(phi) + (1.0 / cos(phi)))) + (_output_height / 2);
                float cyli_col = theta / __PI__ * 512.0 + 512.0;

                FlowData pixel_flow;
                try {
                    pixel_flow = flow_data[int(cyli_row + 2 * _output_height) * _output_width + int(cyli_col)];
                }
                catch (std::exception e) {
                    CGRA_LOGD("ERROR! %f %f", cyli_row, cyli_col);
                    pixel_flow.x = 0;
                    pixel_flow.y = 0;
                }

                float target_row = cyli_row + pixel_flow.y;
                float target_col = cyli_col + pixel_flow.x;
                if (target_col < 0) target_col += 1024;
                if (target_col >= 1024) target_col -= 1024;

                float target_middle = (float(target_row)) - (_output_height / 2);
                float target_phi = 2 * (atan(exp(float(target_middle / (_output_width/2))))) - __PI_2__;
                target_phi = -target_phi;
                float target_theta = ((float(target_col)) / _image_width * 2.0 - 1.0) * __PI__;

                x = cos(target_phi) * sin(target_theta);
                y = sin(target_phi);
                z = cos(target_phi) * cos(target_theta);
                ray = glm::vec4(x, y, -z, 1.0f); // to right hand
                rotation = glm::eulerAngleXYZ(0.0f, 0.0f, glm::radians(-90.0f));
                out = glm::normalize(rotation * ray);
                out = out / out.w;
                x = out.x;
                y = out.y;
                z = out.z;

                z = -z;

                target_phi = atan2(y, sqrt(x * x + z * z));
                target_phi = -target_phi;
                target_theta = atan2(x, z);

                target_row = ((target_phi / __PI_2__) + 1.0) * 0.5 * _image_height;
                target_col = ((target_theta/ __PI__) + 1.0) * 0.5 * _image_width;

                output_flow_data[row * _image_width + col].x = target_col - _col;
                output_flow_data[row * _image_width + col].y = target_row - _row;
                if (output_flow_data[row * _image_width + col].x < -512) output_flow_data[row * _image_width + col].x += 1024;
                if (output_flow_data[row * _image_width + col].x > 512) output_flow_data[row * _image_width + col].x -= 1024;
            }
            else {
                output_flow_data[row * _image_width + col].x = 0.0f;
                output_flow_data[row * _image_width + col].y = 0.0f;
            }
        }
    }
    return output_flow_data;
}

///////////////////////////////////////////

class Cube2Equi
{
public:
	Cube2Equi(uint32_t cubemap_size);

	~Cube2Equi();

	const CubemapsFaceCoord* const getCoord(const uint32_t x, const uint32_t y) const {
		// uint32_t row = y;
		// uint32_t col = x;
		//       return &_map[row * _image_width + col];
		return &_map[y * _image_width + x];
	}

	inline uint32_t getImageWidth() {
		return _image_width;
	}

	inline uint32_t getImageHeight() {
		return _image_height;
	}

private:


	uint32_t _cubemap_size;
	uint32_t _image_width;
	uint32_t _image_height;

	CubemapsFaceCoord* _map;

	CubemapsFace getFaceFromThetaAndPhi(const float& theta, const float& phi);

    void locate(const float axis, const float px, const float py, const float rad, float& outx, float& outy) {
	float sizeRatio = 128.0 / axis;

	outx = sizeRatio * px;
	outy = sizeRatio * py;

	float temp = outx;
	outx = outx * cos(rad) - outy * sin(rad);
	outy = temp * sin(rad) + outy * cos(rad);

	outx += 128.0;
	outy += 128.0;
}
};

Cube2Equi::Cube2Equi(uint32_t cubemap_size)
	: _cubemap_size(cubemap_size)
	, _image_width(cubemap_size * 4) // col j theta
	, _image_height(cubemap_size * 2) // row i phi
	, _map(nullptr)
{
	_map = (CubemapsFaceCoord*)malloc(_image_width * _image_height * sizeof(CubemapsFaceCoord));
	if (!_map) {
		// CGRA_LOGE("_map == nullptr");
	}

	for (uint32_t row = 0; row < _image_height; row++) {
		for (uint32_t col = 0; col < _image_width; col++) {

			// [-1, 1)
			float i = float(row)  / _image_height * 2.0 - 1.0;
			float j = float(col) / _image_width * 2.0 - 1.0;

			// CGRA_LOGD("%d %d %f %f", row, col, i, j);

			// phi [-pi/2, pi/2)
			// theta [-pi, pi)
			float phi = i * __PI_2__;
			float theta = j * __PI__;

			float x = cos(phi) * cos(theta);
			float y = sin(phi);
			float z = cos(phi) * sin(theta);

			_map[row * _image_width + col].face = getFaceFromThetaAndPhi(theta, phi);

			switch (_map[row * _image_width + col].face) {
			case CUBEMAPS_TOP: {
				locate(y, z, x, __PI__, _map[row * _image_width + col].x, _map[row * _image_width + col].y);
				break;
			}
			case CUBEMAPS_DOWN: {
				locate(y, x, z, -__PI_2__, _map[row * _image_width + col].x, _map[row * _image_width + col].y);
				break;
			}
			case CUBEMAPS_LEFT: {
				locate(z, x, y, __PI__, _map[row * _image_width + col].x, _map[row * _image_width + col].y);
				break;
			}
			case CUBEMAPS_RIGHT: {
				locate(z, y, x, __PI_2__, _map[row * _image_width + col].x, _map[row * _image_width + col].y);
				break;
			}
			case CUBEMAPS_FRONT: {
				locate(x, z, y, 0.0, _map[row * _image_width + col].x, _map[row * _image_width + col].y);
				break;
			}
			case CUBEMAPS_BACK: {
				locate(x, y, z, -__PI_2__, _map[row * _image_width + col].x, _map[row * _image_width + col].y);
				break;
			}
			default: {
				break;
			}


			}
		}
	}
}

bool isDoubleInRange(const float & input, const float & bottom, const float & top)
{
	return ((input - bottom) >= 0) && (input < top);
}

CubemapsFace Cube2Equi::getFaceFromThetaAndPhi(const float & theta, const float & phi)
{
	CubemapsFace cubeFaceId;
	float normTheta = 0;

	// Looking at the cube from top down
	// FRONT zone
	if (isDoubleInRange(theta, -__PI_4__, __PI_4__)) {
		cubeFaceId = CUBEMAPS_FRONT;
		normTheta  = theta;
	}
	// LEFT zone
	else if (isDoubleInRange(theta, -(__PI_2__ + __PI_4__), -__PI_4__)) {
		cubeFaceId = CUBEMAPS_LEFT;
		normTheta  = theta + __PI_2__;
	}
	// RIGHT zone
	else if (isDoubleInRange(theta, __PI_4__, __PI_2__ + __PI_4__)) {
		cubeFaceId = CUBEMAPS_RIGHT;
		normTheta  = theta - __PI_2__;
	}
	else {
		cubeFaceId = CUBEMAPS_BACK;

		if (theta > 0.0) {
			normTheta = theta - __PI__;
		}
		else {
			normTheta = theta + __PI__;
		}
	}

	// find out which segment the line strikes to
	float phiThreshold = atan2(__RADIUS__, __RADIUS__ / cos(normTheta));

	// in the top segment
	if (phi > phiThreshold) {
		cubeFaceId = CUBEMAPS_DOWN;
	}
	// in the bottom segment
	else if (phi < -phiThreshold) {
		cubeFaceId = CUBEMAPS_TOP;
	}
	// in the middle segment
	else {
		;
	}

	return cubeFaceId;
}

Cube2Equi::~Cube2Equi()
{
	if (_map) {
		free(_map);
		_map = nullptr;
	}

}

Cube2Equi* _proj_of_c_2_e = new Cube2Equi(256);


FlowData* flow_internel_cubemap_to_equirect(float* front, float* back, float* left, float* right, float* top, float* bottom)
{
    FlowData* front_flow_data;
    front_flow_data = (FlowData*)(front);

    FlowData* back_flow_data;
    back_flow_data = (FlowData*)(back);

    FlowData* left_flow_data;
    left_flow_data = (FlowData*)(left);

    FlowData* right_flow_data;
    right_flow_data = (FlowData*)(right);

    FlowData* top_flow_data;
    top_flow_data = (FlowData*)(top);

    FlowData* bottom_flow_data;
    bottom_flow_data = (FlowData*)(bottom);

    unsigned int panoWidth  = _proj_of_c_2_e->getImageWidth();
    unsigned int panoHeight = _proj_of_c_2_e->getImageHeight();

    const CubemapsFaceCoord* coord = NULL;

    FlowData* optical_flow = (FlowData*)malloc(panoWidth * panoHeight * sizeof(FlowData));
    for (uint32_t i = 0; i < panoWidth; ++i) {
        for (uint32_t j = 0; j < panoHeight; ++j) {

            coord = _proj_of_c_2_e->getCoord(i, j);
            int x = coord->x;
            int y = coord->y;

            // int row = y;
            // int col = x;

            // float theta = 0;
            // float phi = 0;

            // int height = 256;
            // int width = 256;

            switch (coord->face) {
            case (CUBEMAPS_TOP) : {
                optical_flow[j * panoWidth + i] = top_flow_data[y * 256 + x];
                // float flow_x = top_flow_data[row * width + col].x;
                // float flow_y = top_flow_data[row * width + col].y;
                // float x0, y0, x1, y1;
                // convertToRadian(CameraDirection::TOP, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                // convertToRadian(CameraDirection::TOP, 2 * ((float(col) + flow_x) / width - 0.5), 2 * ((float(row) + flow_y) / height - 0.5), x1, y1);
                // theta = x1-x0;
                // phi = y1-y0;
                break;
            }
            case (CUBEMAPS_LEFT) : {
                optical_flow[j * panoWidth + i] = left_flow_data[y * 256 + x];
                // float flow_x = left_flow_data[row * width + col].x;
                // float flow_y = left_flow_data[row * width + col].y;
                // float x0, y0, x1, y1;
                // convertToRadian(CameraDirection::LEFT, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                // convertToRadian(CameraDirection::LEFT, 2 * ((float(col) + flow_x) / width - 0.5), 2 * ((float(row) + flow_y) / height - 0.5), x1, y1);
                // theta = x1-x0;
                // phi = y1-y0;
                break;
            }
            case (CUBEMAPS_FRONT) : {
                optical_flow[j * panoWidth + i] = front_flow_data[y * 256 + x];
                // float flow_x = front_flow_data[row * width + col].x;
                // float flow_y = front_flow_data[row * width + col].y;
                // float x0, y0, x1, y1;
                // convertToRadian(CameraDirection::FRONT, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                // convertToRadian(CameraDirection::FRONT, 2 * ((float(col) + flow_x) / width - 0.5), 2 * ((float(row) + flow_y) / height - 0.5), x1, y1);
                // theta = x1-x0;
                // phi = y1-y0;
                break;
            }
            case (CUBEMAPS_RIGHT) : {
                optical_flow[j * panoWidth + i] = right_flow_data[y * 256 + x];
                // float flow_x = right_flow_data[row * width + col].x;
                // float flow_y = right_flow_data[row * width + col].y;
                // float x0, y0, x1, y1;
                // convertToRadian(CameraDirection::RIGHT, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                // convertToRadian(CameraDirection::RIGHT, 2 * ((float(col) + flow_x) / width - 0.5), 2 * ((float(row) + flow_y) / height - 0.5), x1, y1);
                // theta = x1-x0;
                // phi = y1-y0;
                break;
            }
            case (CUBEMAPS_BACK) : {
                optical_flow[j * panoWidth + i] = back_flow_data[y * 256 + x];
                // float flow_x = back_flow_data[row * width + col].x;
                // float flow_y = back_flow_data[row * width + col].y;
                // float x0, y0, x1, y1;
                // convertToRadian(CameraDirection::BACK, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                // convertToRadian(CameraDirection::BACK, 2 * ((float(col) + flow_x) / width - 0.5), 2 * ((float(row) + flow_y) / height - 0.5), x1, y1);
                // theta = x1-x0;
                // phi = y1-y0;
                break;
            }
            case (CUBEMAPS_DOWN) : {
                optical_flow[j * panoWidth + i] = bottom_flow_data[y * 256 + x];
                // float flow_x = bottom_flow_data[row * width + col].x;
                // float flow_y = bottom_flow_data[row * width + col].y;
                // float x0, y0, x1, y1;
                // convertToRadian(CameraDirection::DOWN, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                // convertToRadian(CameraDirection::DOWN, 2 * ((float(col) + flow_x) / width - 0.5), 2 * ((float(row) + flow_y) / height - 0.5), x1, y1);
                // theta = x1-x0;
                // phi = y1-y0;
                break;
            }
            default: {
                optical_flow[j * panoWidth + i].x = 0;
                optical_flow[j * panoWidth + i].y = 0;

            }

            // if (theta > 512) theta -= 1024;
            // if (theta < -512) theta += 1024;

            // optical_flow[j * panoWidth + i].x = theta;
            // optical_flow[j * panoWidth + i].y = phi;

            }
        }
    }
    return optical_flow;
}

const static int __FRONT__ = 0;
const static int __BACK__ = 1;
const static int __LEFT__ = 2;
const static int __RIGHT__ = 3;
const static int __TOP__ = 4;
const static int __BOTTOM__ = 5;

void internel_convert_cubemap_flow_to_equirect_flow(int face, float* data)
{
    int height = 256;
    int width = 256;

    FlowData* flow_data;
    flow_data = (FlowData*)data;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {

            float x = flow_data[row * width + col].x;
            float y = flow_data[row * width + col].y;

            float theta;
            float phi;

            switch (face) {
            case (__FRONT__) :
            {
                float x0, y0, x1, y1;
                convertToRadian(CameraDirection::FRONT, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                convertToRadian(CameraDirection::FRONT, 2 * ((float(col) + x) / width - 0.5), 2 * ((float(row) + y) / height - 0.5), x1, y1);
                theta = x1-x0;
                phi = y1-y0;
                break;
            }
            case (__BACK__) :
            {
                float x0, y0, x1, y1;
                convertToRadian(CameraDirection::BACK, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                convertToRadian(CameraDirection::BACK, 2 * ((float(col) + x) / width - 0.5), 2 * ((float(row) + y) / height - 0.5), x1, y1);
                theta = x1-x0;
                phi = y1-y0;
                break;
            }
            case (__LEFT__) :
            {
                float x0, y0, x1, y1;
                convertToRadian(CameraDirection::LEFT, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                convertToRadian(CameraDirection::LEFT, 2 * ((float(col) + x) / width - 0.5), 2 * ((float(row) + y) / height - 0.5), x1, y1);
                theta = x1-x0;
                phi = y1-y0;
                break;
            }
            case (__RIGHT__) :
            {
                float x0, y0, x1, y1;
                convertToRadian(CameraDirection::RIGHT, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                convertToRadian(CameraDirection::RIGHT, 2 * ((float(col) + x) / width - 0.5), 2 * ((float(row) + y) / height - 0.5), x1, y1);
                theta = x1-x0;
                phi = y1-y0;
                break;
            }
            case (__TOP__) :
            {
                float x0, y0, x1, y1;
                convertToRadian(CameraDirection::TOP, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                convertToRadian(CameraDirection::TOP, 2 * ((float(col) + x) / width - 0.5), 2 * ((float(row) + y) / height - 0.5), x1, y1);
                theta = x1-x0;
                phi = y1-y0;
                break;
            }
            case (__BOTTOM__) :
            {
                float x0, y0, x1, y1;
                convertToRadian(CameraDirection::DOWN, 2 * (float(col) / width - 0.5), 2 * (float(row) / height - 0.5), x0, y0);
                convertToRadian(CameraDirection::DOWN, 2 * ((float(col) + x) / width - 0.5), 2 * ((float(row) + y) / height - 0.5), x1, y1);
                theta = x1-x0;
                phi = y1-y0;
                break;
            }
            default: {
                theta = 0;
                phi = 0;
            }
            }

            if (theta > 512) theta -= 1024;
            if (theta < -512) theta += 1024;

            flow_data[row * width + col].x = theta;
            flow_data[row * width + col].y = phi;
        }
    }
}


#include <pybind11/numpy.h>

using namespace pybind11;
using namespace pybind11::literals;

array_t<uint8_t> RGB_equirect_to_cylinder(const array_t<uint8_t>& input, const int rows, const int cols, const int channels)
{
    Equirect2Cylinder e2c;

    buffer_info input_buff = input.request();

    CGRA_LOGD("rows:%d, cols:%d, channel:%d", rows, cols, channels);
    auto result = array_t<uint8_t>(e2c.cylinder_rows * e2c.cylinder_cols * channels);
    buffer_info result_buff = result.request();

    e2c.init_input((uint8_t*)input_buff.ptr);
    e2c.generate_cylin_RGB();

    uint8_t* p = (uint8_t*)result_buff.ptr;
    memcpy(p, e2c.cyli_buff, e2c.cylinder_rows * e2c.cylinder_cols * channels * sizeof(uint8_t));

    result.resize({(ssize_t)e2c.cylinder_rows ,(ssize_t)e2c.cylinder_cols, (ssize_t)channels});

    e2c.release_RGB_input();
    return result;
}

array_t<float> flow_equirect_to_cylinder(const array_t<float>& input, const int rows, const int cols, const int channels)
{
    Equirect2Cylinder e2c;

    buffer_info input_buff = input.request();

    CGRA_LOGD("rows:%d, cols:%d, channel:%d", rows, cols, channels);
    auto result = array_t<float>(e2c.cylinder_rows * e2c.cylinder_cols * channels);
    buffer_info result_buff = result.request();

    e2c.init_flow_input((FlowData*)input_buff.ptr);
    e2c.synthesizeCyli();

    float* p = (float*)result_buff.ptr;
    memcpy(p, e2c.cyli_flow_buff, e2c.cylinder_rows * e2c.cylinder_cols * channels * sizeof(float));

    result.resize({(ssize_t)e2c.cylinder_rows ,(ssize_t)e2c.cylinder_cols, (ssize_t)channels});

    e2c.release_flow_input();
    return result;
}

array_t<uint8_t> RGB_equirect_to_cubepadding(const array_t<uint8_t>& input, const int rows, const int cols, const int channels)
{
    Equirect2Cylinder e2c;

    buffer_info input_buff = input.request();

    CGRA_LOGD("rows:%d, cols:%d, channel:%d", rows, cols, channels);
    auto result = array_t<uint8_t>(256*4 * 256*4 * channels);
    buffer_info result_buff = result.request();

    e2c.init_input((uint8_t*)input_buff.ptr);
    e2c.project_RGB_equirect_to_cubemap();

    cv::Mat output_CL(256 * 4, 256 * 4, CV_8UC3, cv::Scalar(0, 0, 0));

    RGB_cubemap_2_cubepadding(e2c.output_RGB_front, e2c.output_RGB_back, e2c.output_RGB_left, e2c.output_RGB_right, e2c.output_RGB_top, e2c.output_RGB_bottom, output_CL);

    uint8_t* p = (uint8_t*)result_buff.ptr;
    memcpy(p, output_CL.data, 256*4 * 256*4 * channels * sizeof(uint8_t));

    result.resize({(ssize_t)256*4 ,(ssize_t)256*4, (ssize_t)channels});

    e2c.release_RGB_input();
    return result;
}

array_t<float> flow_equirect_to_cubepadding(const array_t<float>& input, const int rows, const int cols, const int channels)
{
    Equirect2Cylinder e2c;

    buffer_info input_buff = input.request();

    CGRA_LOGD("rows:%d, cols:%d, channel:%d", rows, cols, channels);
    auto result = array_t<float>(256*4 * 256*4 * channels);
    buffer_info result_buff = result.request();

    e2c.init_flow_input((FlowData*)input_buff.ptr);
    e2c.porject_flow_equirect_to_cubemap();

    FlowData *cl_P_output_optical_flow = (FlowData *)malloc(4 * 4 * 256 * 256 * sizeof(FlowData));

    flow_cubemap_2_cubepadding(e2c.output_flow_front, e2c.output_flow_back, e2c.output_flow_left, e2c.output_flow_right, e2c.output_flow_top, e2c.output_flow_bottom, cl_P_output_optical_flow);

    float* p = (float*)result_buff.ptr;
    memcpy(p, cl_P_output_optical_flow, 256*4 * 256*4 * channels * sizeof(float));

    result.resize({(ssize_t)256*4 ,(ssize_t)256*4, (ssize_t)channels});

    e2c.release_flow_input();
    free(cl_P_output_optical_flow);
    return result;
}

array_t<float> flow_cylinder_to_equirect(const array_t<float>& input, const int rows, const int cols, const int channels)
{
    buffer_info input_buff = input.request();

    CGRA_LOGD("rows:%d, cols:%d, channel:%d", rows, cols, channels);
    auto result = array_t<float>(512 * 1024 * channels);
    buffer_info result_buff = result.request();


    FlowData* temp = flow_internel_cylinder_to_equirect((float*)input_buff.ptr);
    float* p = (float*)result_buff.ptr;
    memcpy(p, temp, 512 * 1024 * channels * sizeof(float));
    result.resize({(ssize_t)512 ,(ssize_t)1024, (ssize_t)channels});

    free(temp);
    return result;
}

array_t<float> flow_cubemap_to_equirect(const array_t<float>& front, 
                                        const array_t<float>& back, 
                                        const array_t<float>& left, 
                                        const array_t<float>& right, 
                                        const array_t<float>& top, 
                                        const array_t<float>& bottom, 
                                        const int rows, 
                                        const int cols, 
                                        const int channels)
{
    buffer_info front_buff = front.request();
    buffer_info back_buff = back.request();
    buffer_info left_buff = left.request();
    buffer_info right_buff = right.request();
    buffer_info top_buff = top.request();
    buffer_info bottom_buff = bottom.request();

    CGRA_LOGD("rows:%d, cols:%d, channel:%d", rows, cols, channels);
    auto result = array_t<float>(512 * 1024 * channels);
    buffer_info result_buff = result.request();

    internel_convert_cubemap_flow_to_equirect_flow(0, (float*)front_buff.ptr);
    internel_convert_cubemap_flow_to_equirect_flow(1, (float*)back_buff.ptr);
    internel_convert_cubemap_flow_to_equirect_flow(2, (float*)left_buff.ptr);
    internel_convert_cubemap_flow_to_equirect_flow(3, (float*)right_buff.ptr);
    internel_convert_cubemap_flow_to_equirect_flow(4, (float*)top_buff.ptr);
    internel_convert_cubemap_flow_to_equirect_flow(5, (float*)bottom_buff.ptr);

    FlowData* temp = flow_internel_cubemap_to_equirect((float*)front_buff.ptr, (float*)back_buff.ptr, (float*)left_buff.ptr, (float*)right_buff.ptr, (float*)top_buff.ptr, (float*)bottom_buff.ptr);
    float* p = (float*)result_buff.ptr;
    memcpy(p, temp, 512 * 1024 * channels * sizeof(float));
    result.resize({(ssize_t)512 ,(ssize_t)1024, (ssize_t)channels});

    free(temp);
    return result;
}



PYBIND11_MODULE(libexample, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("RGB_equirect_to_cylinder", &RGB_equirect_to_cylinder, "A function RGB_equirect_to_cylinder");
    m.def("flow_equirect_to_cylinder", &flow_equirect_to_cylinder, "A function flow_equirect_to_cylinder");
    m.def("RGB_equirect_to_cubepadding", &RGB_equirect_to_cubepadding, "A function RGB_equirect_to_cubepadding");
    m.def("flow_equirect_to_cubepadding", &flow_equirect_to_cubepadding, "A function flow_equirect_to_cubepadding");

    m.def("flow_cylinder_to_equirect", &flow_cylinder_to_equirect, "A function flow_cylinder_to_equirect");
    m.def("flow_cubemap_to_equirect", &flow_cubemap_to_equirect, "A function flow_cubemap_to_equirect");
}