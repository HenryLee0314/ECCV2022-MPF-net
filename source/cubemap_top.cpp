#include "cubemap_top.h"
#include "log.h"

#include <cstdlib>
#include <cmath>
#include <cstdio>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/euler_angles.hpp"

const static float __PI__ = 3.14159265358979323846;
const static float __PI_2__ = 1.57079632679489661923;
const static float __PI_4__ = 0.78539816339744830962;
const static float __RADIUS__ = 1.0;

void locate(const float axis, const float px, const float py, float& outx, float& outy) {
    float sizeRatio = 128.0 / axis;

    outx = sizeRatio * px;
    outy = sizeRatio * py;

    outx += 128.0;
    outy += 128.0;
}

// #define CGRA_LOG(format, ...) printf("%s[%d]%s(): " format "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)


void Cube2Cyli::SetMap(glm::mat4 rotation, float theta, float phi, float& outx, float& outy, CubemapsFace& face, float& outTheta, float& outPhi)
{

    float x = cos(phi) * sin(theta);
    float y = sin(phi);
    float z = cos(phi) * cos(theta);

    // left hand to right hand
	z = -z;

    glm::vec4 vec(x,y,z, 1.0f);
    vec = glm::normalize(rotation * vec);
	vec = vec / vec.w;
    x = vec.x;
    y = vec.y;
    z = vec.z;

    // right hand to left hand
	z = -z;
	phi = atan2(y, sqrt(x * x + z * z));
    theta = atan2(x, z);


    outTheta = theta;
	outPhi = phi;

    face = getFaceFromThetaAndPhi(theta, phi);
	
    switch (face) {
    case CUBEMAPS_TOP: {
		locate(y, x, -(-z), outx, outy);
        break;
    }
    case CUBEMAPS_DOWN: {
		locate(-y, x, -z, outx, outy);
        break;
    }
    case CUBEMAPS_LEFT: {
		locate(-x, z, -y, outx, outy);
        break;
    }
    case CUBEMAPS_RIGHT: {
		locate(x, -z, -y, outx, outy);
        break;
    }
    case CUBEMAPS_FRONT: {
		locate(z, x, -y, outx, outy);
        break;
    }
    case CUBEMAPS_BACK: {
		locate(-z, -x, -y, outx, outy);
        break;
    }
    default: {
        break;
    }
    }
}


Cube2Cyli::Cube2Cyli(uint32_t cubemap_size, float degree)
    : _degree(degree)
    , _cubemap_size(cubemap_size)
    , _image_width(cubemap_size * 4) // col j theta
    , _image_height(cubemap_size * 2) // row i phi
    , _map_front(nullptr)
    , _map_equirect(nullptr)
    , _map_top(nullptr)
    , _map_left(nullptr)
{


    if (!_map_front || !_map_top || !_map_left) {
        // CGRA_LOGE("_map == nullptr");
    }
    _degree = abs(_degree) < 89.9 ? _degree : 89.9;
    _output_width = _image_width;
    float rand = _degree / 180.0 * __PI__;
    _output_height = uint32_t((_output_width/2) * log(tan(rand) + (1 / cos(rand))));
    _output_height = _output_height * 2;
	CGRA_LOGD("%d", _output_height);

    _map_equirect = (CubemapsFaceCoord*)malloc(_image_width * _image_height * sizeof(CubemapsFaceCoord));

	_map_front = (CubemapsFaceCoord*)malloc(_output_width * _output_height * sizeof(CubemapsFaceCoord));
    _map_top = (CubemapsFaceCoord*)malloc(_output_width * _output_height * sizeof(CubemapsFaceCoord));
    _map_left = (CubemapsFaceCoord*)malloc(_output_width * _output_height * sizeof(CubemapsFaceCoord));

	cyli_front = (CubemapsFaceCoordThetaPhi*)malloc(_output_width * _output_height * sizeof(CubemapsFaceCoordThetaPhi));
	cyli_top = (CubemapsFaceCoordThetaPhi*)malloc(_output_width * _output_height * sizeof(CubemapsFaceCoordThetaPhi));
	cyli_left = (CubemapsFaceCoordThetaPhi*)malloc(_output_width * _output_height * sizeof(CubemapsFaceCoordThetaPhi));

	// cyli_front_to_equirect = (CubemapsFaceCoordThetaPhi*)malloc(_output_width * _output_height * sizeof(CubemapsFaceCoordThetaPhi));
	// cyli_top_to_equirect = (CubemapsFaceCoordThetaPhi*)malloc(_output_width * _output_height * sizeof(CubemapsFaceCoordThetaPhi));
	// cyli_left_to_equirect = (CubemapsFaceCoordThetaPhi*)malloc(_output_width * _output_height * sizeof(CubemapsFaceCoordThetaPhi));


	for (uint32_t row = 0; row < _image_height; row++) {
        for (uint32_t col = 0; col < _image_width; col++) {
			float theta = ((float(col) + 0.5f) / _image_width * 2.0 - 1.0) * __PI__;
			float phi = ((float(row) + 0.5f) / _image_height * 2.0 - 1.0) * __PI_2__;
			phi = -phi;


			float _, __;

			SetMap(glm::eulerAngleXYZ(0.0f, 0.0f, 0.0f), theta, phi, _map_equirect[row * _output_width + col].x, _map_equirect[row * _output_width + col].y, _map_equirect[row * _output_width + col].face, _, __);
		}
	}

    for (uint32_t row = 0; row < _output_height; row++) {
        for (uint32_t col = 0; col < _output_width; col++) {

            float middle = (float(row) + 0.5) - (_output_height / 2);
            float phi = 2 * (atan(exp(float(middle / (_output_width/2))))) - __PI_2__;

			float theta = ((float(col) + 0.5f) / _image_width * 2.0 - 1.0) * __PI__;
			phi = -phi;

			SetMap(glm::eulerAngleXYZ(0.0f, 0.0f, 0.0f), theta, phi, _map_front[row * _output_width + col].x, _map_front[row * _output_width + col].y, _map_front[row * _output_width + col].face, cyli_front[row * _output_width + col].theta, cyli_front[row * _output_width + col].phi);
			SetMap(glm::eulerAngleXYZ(glm::radians(90.0f), 0.0f, 0.0f),theta, phi, _map_top[row * _output_width + col].x, _map_top[row * _output_width + col].y, _map_top[row * _output_width + col].face, cyli_top[row * _output_width + col].theta, cyli_top[row * _output_width + col].phi);
			SetMap(glm::eulerAngleXYZ(0.0f, 0.0f, glm::radians(-90.0f)), theta, phi, _map_left[row * _output_width + col].x, _map_left[row * _output_width + col].y, _map_left[row * _output_width + col].face, cyli_left[row * _output_width + col].theta, cyli_left[row * _output_width + col].phi);

        }
    }
}

bool isfloatInRange(const float & input, const float & bottom, const float & top)
{
    return ((input - bottom) >= 0) && (input < top);
}

CubemapsFace Cube2Cyli::getFaceFromThetaAndPhi(const float & theta, const float & phi)
{
    CubemapsFace cubeFaceId;
    float normTheta = 0;

    if (isfloatInRange(theta, -__PI_4__, __PI_4__)) {
        cubeFaceId = CUBEMAPS_FRONT;
        normTheta  = theta;
    }
    else if (isfloatInRange(theta, -(__PI_2__ + __PI_4__), -__PI_4__)) {
        cubeFaceId = CUBEMAPS_LEFT;
        normTheta  = theta + __PI_2__;
    }
    else if (isfloatInRange(theta, __PI_4__, __PI_2__ + __PI_4__)) {
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

    float phiThreshold = atan2(__RADIUS__, __RADIUS__ / cos(normTheta));

    if (phi > phiThreshold) {
		cubeFaceId = CUBEMAPS_TOP;

    }
    else if (phi < -phiThreshold) {
		cubeFaceId = CUBEMAPS_DOWN;

    }
    else {
        ;
    }

    return cubeFaceId;
}

Cube2Cyli::~Cube2Cyli()
{
    if (_map_front) {
        free(_map_front);
        _map_front = nullptr;
    }

}

