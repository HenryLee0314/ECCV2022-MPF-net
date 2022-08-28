#ifndef CUBEMAPS_TO_CYLINDRICAL_PROJECTION_H
#define CUBEMAPS_TO_CYLINDRICAL_PROJECTION_H

#include <cstdint>
#include <glm/glm.hpp>
// #include "cubemaps_to_equirectangular_projection.h"

enum CubemapsFace {
	CUBEMAPS_TOP = 0,
	CUBEMAPS_LEFT,
	CUBEMAPS_FRONT,
	CUBEMAPS_RIGHT,
	CUBEMAPS_BACK,
	CUBEMAPS_DOWN,
	CUBEMAPS_FACE_NUM
};

struct CubemapsFaceCoord {
	CubemapsFace face;    // the face of the cube
	float     x;       // the x coordinate
	float     y;       // the y coordinate
};

struct CubemapsFaceCoordThetaPhi {
	float     theta;       // the x coordinate
	float     phi;       // the y coordinate
};

void locate(const float axis, const float px, const float py, float& outx, float& outy);

class Cube2Cyli
{
public:
	Cube2Cyli(uint32_t cubemap_size, float degree = 90.0);

	~Cube2Cyli();

	const CubemapsFaceCoord* const getCoord_front(const uint32_t x, const uint32_t y) const {
		return &_map_front[y * _image_width + x];
	}

    const CubemapsFaceCoord* const getCoord_top(const uint32_t x, const uint32_t y) const {
		return &_map_top[y * _image_width + x];
	}

    const CubemapsFaceCoord* const getCoord_left(const uint32_t x, const uint32_t y) const {
		return &_map_left[y * _image_width + x];
	}

    const CubemapsFaceCoord* const getCoordEquirect(const uint32_t x, const uint32_t y) const {
		// uint32_t row = y;
		// uint32_t col = x;
		//       return &_map[row * _image_width + col];
		return &_map_equirect[y * _output_width + x];
	}

	inline uint32_t getImageWidth() {
		return _image_width;
	}

	inline uint32_t getImageHeight() {
		return _image_height;
	}

    inline uint32_t getOutputImageWidth() {
		return _output_width;
	}

	inline uint32_t getOutputImageHeight() {
		return _output_height;
	}

	const CubemapsFaceCoordThetaPhi* const getCoord_theta_phi_front(const uint32_t x, const uint32_t y) const {
		return &cyli_front[y * _image_width + x];
	}

    const CubemapsFaceCoordThetaPhi* const getCoord_theta_phi_top(const uint32_t x, const uint32_t y) const {
		return &cyli_top[y * _image_width + x];
	}

    const CubemapsFaceCoordThetaPhi* const getCoord_theta_phi_left(const uint32_t x, const uint32_t y) const {
		return &cyli_left[y * _image_width + x];
	}

public:
	void SetMap(glm::mat4 rotation, float theta, float phi, float& outx, float& outy,  CubemapsFace& face, float& outTheta, float& outPhi);

    float    _degree;
	uint32_t _cubemap_size;
	uint32_t _image_width;
	uint32_t _image_height;

	CubemapsFaceCoord* _map_front;
    CubemapsFaceCoord* _map_top;
    CubemapsFaceCoord* _map_left;


    CubemapsFaceCoord* _map_equirect;

	CubemapsFaceCoordThetaPhi* cyli_front;
	CubemapsFaceCoordThetaPhi* cyli_top;
	CubemapsFaceCoordThetaPhi* cyli_left;

    uint32_t _output_width;
	uint32_t _output_height;


	CubemapsFace getFaceFromThetaAndPhi(const float& theta, const float& phi);
};

#endif // CUBEMAPS_TO_CYLINDRICAL_PROJECTION_H