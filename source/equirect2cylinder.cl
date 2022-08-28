

typedef struct CubemapsFaceCoordThetaPhi_s {
	float     theta;       // the x coordinate
	float     phi;       // the y coordinate
}CubemapsFaceCoordThetaPhi;


CubemapsFaceCoordThetaPhi getCoord_theta_phi(CubemapsFaceCoordThetaPhi* map, const int x, const int y)
{
    int _image_width = 1024;
	return map[y * _image_width + x];
}


__kernel void RGB(__global CubemapsFaceCoordThetaPhi *front, __global CubemapsFaceCoordThetaPhi *top, __global CubemapsFaceCoordThetaPhi *left, __global char* output, __global char* input, int width, int single_height)
{
    const int index = get_global_id(0);
    int cylinder_row = index / width;
    int cylinder_col = index % width;

    float __PI__ = 3.1415926535894932;
    float __PI_2__ = __PI__ / 2;

    int getImageWidth = 1024;
    int getImageHeight = 512;

    CubemapsFaceCoordThetaPhi coord_theta_phi;

    if (cylinder_row < single_height) {
        // front
        coord_theta_phi = getCoord_theta_phi(front, cylinder_col, cylinder_row);
    }
    else if (cylinder_row < 2 * single_height) {
        // top
        coord_theta_phi = getCoord_theta_phi(top, cylinder_col, cylinder_row - single_height);
    }
    else {
        // left
        coord_theta_phi = getCoord_theta_phi(left, cylinder_col, cylinder_row - 2 * single_height);
    }

    float col = ((coord_theta_phi.theta / __PI__ + 1.0) / 2.0) * getImageWidth;
    float row = ((-coord_theta_phi.phi / __PI_2__ + 1.0) / 2.0) * getImageHeight;

    output[3 * index + 0] = input[3 * ((int)row * getImageWidth + (int)col) + 0];
    output[3 * index + 1] = input[3 * ((int)row * getImageWidth + (int)col) + 1];
    output[3 * index + 2] = input[3 * ((int)row * getImageWidth + (int)col) + 2];
}

typedef struct FlowData_s
{
    float x;
    float y;
}FlowData;


__kernel void flow(__global CubemapsFaceCoordThetaPhi *front, __global CubemapsFaceCoordThetaPhi *top, __global CubemapsFaceCoordThetaPhi *left, __global FlowData* output, __global FlowData* input, int width, int single_height)
{
    const int index = get_global_id(0);
    int cylinder_row = index / width;
    int cylinder_col = index % width;

    float __PI__ = 3.1415926535894932;
    float __PI_2__ = __PI__ / 2;

  
    int _image_width = 1024;
    int _image_height = 512;
    int _output_width = 1024;
    int _output_height = single_height;

    CubemapsFaceCoordThetaPhi coord_theta_phi;
    if (cylinder_row < single_height) {
        // front
        // coord_theta_phi = getCoord_theta_phi(front, cylinder_col, cylinder_row);
        int i = cylinder_col;
        int j = cylinder_row;

        float middle = ((float)(j) + 0.5) - (_output_height / 2); // (-h/2, h/2)
        float original_phi = 2 * (atan(exp((float)(middle / (_output_width / 2))))) - __PI_2__;
        original_phi = (original_phi / __PI_2__ + 1.0) / 2.0 * _image_height;

        float original_theta = i + 0.5f;

        // get flow address position
        coord_theta_phi = getCoord_theta_phi(front, i, j);
        float col = ((coord_theta_phi.theta / __PI__ + 1.0) / 2.0) * _image_width;
        float row = ((-coord_theta_phi.phi / __PI_2__ + 1.0) / 2.0) * _image_height;

        // bi-interp
        float bi_x;
        float bi_y;

        bi_x = input[(int)row * _image_width + (int)col].x;
        bi_y = input[(int)row * _image_width + (int)col].y;

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
        float target_row = ((_output_width / 2.0) * log(tan(new_phi) + (1.0 / cos(new_phi)))) + (_output_height / 2);
        // CGRA_LOG("new_phi:%f target_row:%f", new_phi, target_row);

        float target_col = new_theta / __PI__ * 512.0 + 512.0;

        // CGRA_LOG("-> pos: %f %f ", target_col, target_row);

        // CGRA_LOG("=======");
        output[j * _image_width + i].x = target_col - i;
        output[j * _image_width + i].y = target_row - j;

        if (output[j * _image_width + i].x < -512)
            output[j * _image_width + i].x += 1024;
        if (output[j * _image_width + i].x > 512)
            output[j * _image_width + i].x -= 1024;
    }
    else if (cylinder_row < 2 * single_height) {
        // top
        // coord_theta_phi = getCoord_theta_phi(top, cylinder_col, cylinder_row - single_height);
        int i = cylinder_col;
        int j = cylinder_row - single_height;
        // CGRA_LOG("=======");
        float middle = ((float)j + 0.5) - (_output_height / 2); // (-h/2, h/2)
        float original_phi = 2 * (atan(exp((float)(middle / (_output_width / 2))))) - __PI_2__;
        // original_phi = (original_phi / __PI_2__ + 1.0) / 2.0 * _image_height;
        float original_theta = (((float)i + 0.5f) / _image_width * 2.0 - 1.0) * __PI__;
        float original_x = cos(original_phi) * sin(original_theta);
        float original_y = sin(original_phi);
        float original_z = cos(original_phi) * cos(original_theta);
        // glm::vec4 original_vec(original_x, original_y, original_z, 1.0f);
        // original_vec = glm::normalize(glm::eulerAngleXYZ(glm::radians(90.0f), 0.0f, 0.0f) * original_vec);
        // original_vec = original_vec / original_vec.w;
        // original_x = original_vec.x;
        // original_y = original_vec.y;
        // original_z = original_vec.z;
        float swap = original_y;
        original_y = -original_z;
        original_z = swap;

        original_phi = atan2(original_y, sqrt(original_x * original_x + original_z * original_z));
        original_theta = atan2(original_x, original_z);
        original_phi = ((original_phi / __PI_2__ + 1.0) / 2.0) * _image_height;
        original_theta = ((original_theta / __PI__ + 1.0) / 2.0) * _image_width;

        // get flow address position
        coord_theta_phi = getCoord_theta_phi(top, i, j);
        float col = ((coord_theta_phi.theta / __PI__ + 1.0) / 2.0) * _image_width;
        float row = ((-coord_theta_phi.phi / __PI_2__ + 1.0) / 2.0) * _image_height;

        // bi-interp
        float bi_x;
        float bi_y;

        bi_x = input[(int)row * _image_width + (int)col].x;
        bi_y = input[(int)row * _image_width + (int)col].y;

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
        // glm::vec4 vec(x, y, z, 1.0f);
        // vec = glm::normalize(glm::eulerAngleXYZ(glm::radians(90.0f), 0.0f, 0.0f) * vec);
        // vec = vec / vec.w;
        // x = vec.x;
        // y = vec.y;
        // z = vec.z;
        swap = y;
        y = -z;
        z = swap;

        // to left hand
        z = -z;

        // to theta and phi
        float new_phi = atan2(y, sqrt(x * x + z * z));
        float new_theta = atan2(x, z);
        // CGRA_LOG("-> new theta phi: %f %f", new_theta, new_phi);

        // to Cyli
        float target_row = ((_output_width / 2.0) * log(tan(new_phi) + (1.0 / cos(new_phi)))) + (_output_height / 2);
        // CGRA_LOG("new_phi:%f target_row:%f", new_phi, target_row);

        float target_col = new_theta / __PI__ * 512.0 + 512.0;

        // CGRA_LOG("-> pos: %f %f ", target_col, target_row);

        // CGRA_LOG("=======");
        output[cylinder_row * _image_width + i].x = target_col - i;
        output[cylinder_row * _image_width + i].y = target_row - j;

        if (output[cylinder_row * _image_width + i].x < -512)
            output[cylinder_row * _image_width + i].x += 1024;
        if (output[cylinder_row * _image_width + i].x > 512)
            output[cylinder_row * _image_width + i].x -= 1024;
    }
    else {
        // left
        // coord_theta_phi = getCoord_theta_phi(left, cylinder_col, cylinder_row - 2 * single_height);
        int i = cylinder_col;
        int j = cylinder_row - 2 * single_height;
        // CGRA_LOG("=======");
        float middle = ((float)(j) + 0.5) - (_output_height / 2); // (-h/2, h/2)
        float original_phi = 2 * (atan(exp((float)(middle / (_output_width / 2))))) - __PI_2__;
        original_phi = -original_phi;
        // original_phi = (original_phi / __PI_2__ + 1.0) / 2.0 * _image_height;
        float original_theta = (((float)i + 0.5f) / _image_width * 2.0 - 1.0) * __PI__;
        float original_x = cos(original_phi) * sin(original_theta);
        float original_y = sin(original_phi);
        float original_z = cos(original_phi) * cos(original_theta);
        // glm::vec4 original_vec(original_x, original_y, original_z, 1.0f);
        // original_vec = glm::normalize(glm::eulerAngleXYZ(0.0f, 0.0f, glm::radians(-90.0f)) * original_vec);
        // original_vec = original_vec / original_vec.w;
        // original_x = original_vec.x;
        // original_y = original_vec.y;
        // original_z = original_vec.z;
        float swap = original_x;
        original_x = original_y;
        original_y = -swap;

        original_phi = atan2(original_y, sqrt(original_x * original_x + original_z * original_z));
        original_phi = -original_phi;
        original_theta = atan2(original_x, original_z);
        original_phi = ((original_phi / __PI_2__ + 1.0) / 2.0) * _image_height;
        original_theta = ((original_theta / __PI__ + 1.0) / 2.0) * _image_width;

        // get flow address position
        coord_theta_phi = getCoord_theta_phi(left, i, j);
        float col = ((coord_theta_phi.theta / __PI__ + 1.0) / 2.0) * _image_width;
        float row = ((-coord_theta_phi.phi / __PI_2__ + 1.0) / 2.0) * _image_height;

        // bi-interp
        float bi_x;
        float bi_y;

        bi_x = input[(int)row * _image_width + (int)col].x;
        bi_y = input[(int)row * _image_width + (int)col].y;

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
        // glm::vec4 vec(x, y, z, 1.0f);
        // vec = glm::normalize(glm::eulerAngleXYZ(0.0f, 0.0f, glm::radians(-90.0f)) * vec);
        // vec = vec / vec.w;
        // x = vec.x;
        // y = vec.y;
        // z = vec.z;
        swap = x;
        x = y;
        y = -swap;

        // to left hand
        z = -z;

        // to theta and phi
        float new_phi = atan2(y, sqrt(x * x + z * z));
        float new_theta = atan2(x, z);
        // CGRA_LOG("-> new theta phi: %f %f", new_theta, new_phi);

        // to Cyli
        float target_row = ((_output_width / 2.0) * log(tan(new_phi) + (1.0 / cos(new_phi)))) + (_output_height / 2);
        // CGRA_LOG("new_phi:%f target_row:%f", new_phi, target_row);

        float target_col = new_theta / __PI__ * 512.0 + 512.0;

        // CGRA_LOG("-> pos: %f %f ", target_col, target_row);

        // CGRA_LOG("=======");
        output[cylinder_row * _image_width + i].x = target_col - i;
        output[cylinder_row * _image_width + i].y = target_row - j;

        if (output[cylinder_row * _image_width + i].x < -512)
            output[cylinder_row * _image_width + i].x += 1024;
        if (output[cylinder_row * _image_width + i].x > 512)
            output[cylinder_row * _image_width + i].x -= 1024;
    }

}








void convertToRadian(int face, float x, float y, float *theta, float *phi)
{
    float __PI__ = 3.1415926535897932;

    float z = 1;

    float xp;
    float yp;
    float zp;

    if (face == 0) {
        // TOP
        xp = x;
        yp = -z;
        zp = y;

    } else if (face == 1) {
        // LEFT
        xp = -z;
        yp = y;
        zp = x;

    } else if (face == 2) {
        // FRONT
        xp = x;
        yp = y;
        zp = z;

    } else if (face == 3) {
        // RIGHT
        xp = z;
        yp = y;
        zp = -x;

    } else if (face == 4) {
        // BACK
        xp = -x;
        yp = y;
        zp = -z;
    } else {
        // DOWN
        xp = x;
        yp = z;
        zp = -y;
    }
    
    float temp_theta = atan2(xp, zp);
    float temp_phi = atan2(yp, sqrt(xp * xp + zp * zp));

    (*theta) = temp_theta / __PI__ * 512 + 512;
    (*phi) = temp_phi / (__PI__ / 2) * 256 + 256;

}





__kernel void e2p_RGB(__global char* input, __global char* output_top, __global char* output_left, __global char* output_front, __global char* output_right, __global char* output_back, __global char* output_bottom)
{
    int width = 256;
    int height = 256;
    const int index = get_global_id(0);
    int row = index / 256;
    int col = index % 256;

    float theta;
    float phi;
    float x = 2 * ((float)col / width - 0.5);
    float y = 2 * ((float)row / height - 0.5);
    float z = 1;
    float __PI__ = 3.1415926535897932;
    float xp;
    float yp;
    float zp;
    

    // TOP
    xp = x;
    yp = -z;
    zp = y;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    output_top[3 * index + 0] = input[3 * ((int)phi * 1024 + (int)theta) + 0];
    output_top[3 * index + 1] = input[3 * ((int)phi * 1024 + (int)theta) + 1];
    output_top[3 * index + 2] = input[3 * ((int)phi * 1024 + (int)theta) + 2];


    // LEFT
    xp = -z;
    yp = y;
    zp = x;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    output_left[3 * index + 0] = input[3 * ((int)phi * 1024 + (int)theta) + 0];
    output_left[3 * index + 1] = input[3 * ((int)phi * 1024 + (int)theta) + 1];
    output_left[3 * index + 2] = input[3 * ((int)phi * 1024 + (int)theta) + 2];


    // FRONT
    xp = x;
    yp = y;
    zp = z;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    output_front[3 * index + 0] = input[3 * ((int)phi * 1024 + (int)theta) + 0];
    output_front[3 * index + 1] = input[3 * ((int)phi * 1024 + (int)theta) + 1];
    output_front[3 * index + 2] = input[3 * ((int)phi * 1024 + (int)theta) + 2];


    // RIGHT
    xp = z;
    yp = y;
    zp = -x;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    output_right[3 * index + 0] = input[3 * ((int)phi * 1024 + (int)theta) + 0];
    output_right[3 * index + 1] = input[3 * ((int)phi * 1024 + (int)theta) + 1];
    output_right[3 * index + 2] = input[3 * ((int)phi * 1024 + (int)theta) + 2];


    // BACK
    xp = -x;
    yp = y;
    zp = -z;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    output_back[3 * index + 0] = input[3 * ((int)phi * 1024 + (int)theta) + 0];
    output_back[3 * index + 1] = input[3 * ((int)phi * 1024 + (int)theta) + 1];
    output_back[3 * index + 2] = input[3 * ((int)phi * 1024 + (int)theta) + 2];

    // DOWN
    xp = x;
    yp = z;
    zp = -y;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    output_bottom[3 * index + 0] = input[3 * ((int)phi * 1024 + (int)theta) + 0];
    output_bottom[3 * index + 1] = input[3 * ((int)phi * 1024 + (int)theta) + 1];
    output_bottom[3 * index + 2] = input[3 * ((int)phi * 1024 + (int)theta) + 2];
}


__kernel void e2p_flow(__global FlowData* input, __global FlowData* output_top,__global FlowData* output_left, __global FlowData* output_front, __global FlowData* output_right, __global FlowData* output_back, __global FlowData* output_bottom)
{
    float __PI__ = 3.1415926535897932;
    int width = 256;
    int height = 256;
    const int index = get_global_id(0);
    int row = index / 256;
    int col = index % 256;

    float theta;
    float phi;
    float x = 2 * ((float)col / width - 0.5);
    float y = 2 * ((float)row / height - 0.5);
    float z = 1;
    float xp;
    float yp;
    float zp;


    // TOP
    xp = x;
    yp = -z;
    zp = y;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    float flow_x = input[(int)phi * 1024 + (int)theta].x;
    float flow_y = input[(int)phi * 1024 + (int)theta].y;
    float tar_col = flow_x + theta;
    float tar_row = flow_y + phi;

    if (tar_row < 0) tar_row = 0;
    if (tar_row >= 512) tar_row = 512;
    if (tar_col < 0) tar_col += 1024;
    if (tar_col >= 1024) tar_col -= 1024;

    float tar_theta = ((tar_col / 1024) * 2 - 1) * __PI__;
    float tar_phi = ((tar_row / 512) * 2 - 1) * (__PI__ / 2);

    float tar_x = cos(tar_phi) * sin(tar_theta);
    float tar_y = sin(tar_phi);
    float tar_z = cos(tar_phi) * cos(tar_theta);

    float outx = 0;
    float outy = 0;
    tar_z = -tar_z;

    xp = tar_x;
    yp = -tar_z;
    zp = tar_y;

    tar_x = xp;
    tar_y = yp;
    tar_z = -zp;

    float ratio = 128.0 / tar_z;
    float rad = 0;

    outx = ratio * tar_x;
    outy = ratio * tar_y;

    float temp = outx;
    outx = outx * cos(rad) - outy * sin(rad);
    outy = temp * sin(rad) + outy * cos(rad);

    outx = outx - (col - 128);
    outy = outy - (row - 128);

    output_top[row * 256 + col].x = outx;
    output_top[row * 256 + col].y = outy;

    // LEFT
    xp = -z;
    yp = y;
    zp = x;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    flow_x = input[(int)phi * 1024 + (int)theta].x;
    flow_y = input[(int)phi * 1024 + (int)theta].y;
    tar_col = flow_x + theta;
    tar_row = flow_y + phi;

    if (tar_row < 0) tar_row = 0;
    if (tar_row >= 512) tar_row = 512;
    if (tar_col < 0) tar_col += 1024;
    if (tar_col >= 1024) tar_col -= 1024;

    tar_theta = ((tar_col / 1024) * 2 - 1) * __PI__;
    tar_phi = ((tar_row / 512) * 2 - 1) * (__PI__ / 2);

    tar_x = cos(tar_phi) * sin(tar_theta);
    tar_y = sin(tar_phi);
    tar_z = cos(tar_phi) * cos(tar_theta);

    outx = 0;
    outy = 0;
    tar_z = -tar_z;

    xp = -tar_z;
    yp = tar_y;
    zp = tar_x;

    tar_x = xp;
    tar_y = yp;
    tar_z = -zp;

    ratio = 128.0 / tar_z;
    rad = 0;

    outx = ratio * tar_x;
    outy = ratio * tar_y;

    temp = outx;
    outx = outx * cos(rad) - outy * sin(rad);
    outy = temp * sin(rad) + outy * cos(rad);

    outx = outx - (col - 128);
    outy = outy - (row - 128);

    output_left[row * 256 + col].x = outx;
    output_left[row * 256 + col].y = outy;

    // FRONT
    xp = x;
    yp = y;
    zp = z;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    flow_x = input[(int)phi * 1024 + (int)theta].x;
    flow_y = input[(int)phi * 1024 + (int)theta].y;
    tar_col = flow_x + theta;
    tar_row = flow_y + phi;

    if (tar_row < 0) tar_row = 0;
    if (tar_row >= 512) tar_row = 512;
    if (tar_col < 0) tar_col += 1024;
    if (tar_col >= 1024) tar_col -= 1024;

    tar_theta = ((tar_col / 1024) * 2 - 1) * __PI__;
    tar_phi = ((tar_row / 512) * 2 - 1) * (__PI__ / 2);

    tar_x = cos(tar_phi) * sin(tar_theta);
    tar_y = sin(tar_phi);
    tar_z = cos(tar_phi) * cos(tar_theta);

    outx = 0;
    outy = 0;
    tar_z = -tar_z;

    xp = tar_x;
    yp = tar_y;
    zp = tar_z;

    tar_x = xp;
    tar_y = yp;
    tar_z = -zp;

    ratio = 128.0 / tar_z;
    rad = 0;

    outx = ratio * tar_x;
    outy = ratio * tar_y;

    temp = outx;
    outx = outx * cos(rad) - outy * sin(rad);
    outy = temp * sin(rad) + outy * cos(rad);

    outx = outx - (col - 128);
    outy = outy - (row - 128);

    output_front[row * 256 + col].x = outx;
    output_front[row * 256 + col].y = outy;

    // RIGHT
    xp = z;
    yp = y;
    zp = -x;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    flow_x = input[(int)phi * 1024 + (int)theta].x;
    flow_y = input[(int)phi * 1024 + (int)theta].y;
    tar_col = flow_x + theta;
    tar_row = flow_y + phi;

    if (tar_row < 0) tar_row = 0;
    if (tar_row >= 512) tar_row = 512;
    if (tar_col < 0) tar_col += 1024;
    if (tar_col >= 1024) tar_col -= 1024;

    tar_theta = ((tar_col / 1024) * 2 - 1) * __PI__;
    tar_phi = ((tar_row / 512) * 2 - 1) * (__PI__ / 2);

    tar_x = cos(tar_phi) * sin(tar_theta);
    tar_y = sin(tar_phi);
    tar_z = cos(tar_phi) * cos(tar_theta);

    outx = 0;
    outy = 0;
    tar_z = -tar_z;

    xp = tar_z;
    yp = tar_y;
    zp = -tar_x;

    tar_x = xp;
    tar_y = yp;
    tar_z = -zp;

    ratio = 128.0 / tar_z;
    rad = 0;

    outx = ratio * tar_x;
    outy = ratio * tar_y;

    temp = outx;
    outx = outx * cos(rad) - outy * sin(rad);
    outy = temp * sin(rad) + outy * cos(rad);

    outx = outx - (col - 128);
    outy = outy - (row - 128);

    output_right[row * 256 + col].x = outx;
    output_right[row * 256 + col].y = outy;

    // BACK
    xp = -x;
    yp = y;
    zp = -z;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    flow_x = input[(int)phi * 1024 + (int)theta].x;
    flow_y = input[(int)phi * 1024 + (int)theta].y;
    tar_col = flow_x + theta;
    tar_row = flow_y + phi;

    if (tar_row < 0) tar_row = 0;
    if (tar_row >= 512) tar_row = 512;
    if (tar_col < 0) tar_col += 1024;
    if (tar_col >= 1024) tar_col -= 1024;

    tar_theta = ((tar_col / 1024) * 2 - 1) * __PI__;
    tar_phi = ((tar_row / 512) * 2 - 1) * (__PI__ / 2);

    tar_x = cos(tar_phi) * sin(tar_theta);
    tar_y = sin(tar_phi);
    tar_z = cos(tar_phi) * cos(tar_theta);

    outx = 0;
    outy = 0;
    tar_z = -tar_z;

    xp = -tar_x;
    yp = tar_y;
    zp = -tar_z;

    tar_x = xp;
    tar_y = yp;
    tar_z = -zp;

    ratio = 128.0 / tar_z;
    rad = 0;

    outx = ratio * tar_x;
    outy = ratio * tar_y;

    temp = outx;
    outx = outx * cos(rad) - outy * sin(rad);
    outy = temp * sin(rad) + outy * cos(rad);

    outx = outx - (col - 128);
    outy = outy - (row - 128);

    output_back[row * 256 + col].x = outx;
    output_back[row * 256 + col].y = outy;

    // DOWN
    xp = x;
    yp = z;
    zp = -y;
    theta = atan2(xp, zp);
    phi = atan2(yp, sqrt(xp * xp + zp * zp));

    theta = theta / __PI__ * 512 + 512;
    phi = phi / (__PI__ / 2) * 256 + 256;

    flow_x = input[(int)phi * 1024 + (int)theta].x;
    flow_y = input[(int)phi * 1024 + (int)theta].y;
    tar_col = flow_x + theta;
    tar_row = flow_y + phi;

    if (tar_row < 0) tar_row = 0;
    if (tar_row >= 512) tar_row = 512;
    if (tar_col < 0) tar_col += 1024;
    if (tar_col >= 1024) tar_col -= 1024;

    tar_theta = ((tar_col / 1024) * 2 - 1) * __PI__;
    tar_phi = ((tar_row / 512) * 2 - 1) * (__PI__ / 2);

    tar_x = cos(tar_phi) * sin(tar_theta);
    tar_y = sin(tar_phi);
    tar_z = cos(tar_phi) * cos(tar_theta);

    outx = 0;
    outy = 0;
    tar_z = -tar_z;

    xp = tar_x;
    yp = tar_z;
    zp = -tar_y;

    tar_x = xp;
    tar_y = yp;
    tar_z = -zp;

    ratio = 128.0 / tar_z;
    rad = 0;

    outx = ratio * tar_x;
    outy = ratio * tar_y;

    temp = outx;
    outx = outx * cos(rad) - outy * sin(rad);
    outy = temp * sin(rad) + outy * cos(rad);

    outx = outx - (col - 128);
    outy = outy - (row - 128);

    output_bottom[row * 256 + col].x = outx;
    output_bottom[row * 256 + col].y = outy;
}