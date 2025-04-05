#ifndef DILATION_H
#define DILATION_H

#ifdef __cplusplus
#include <iostream>
#include <vector>
#include "../image.h"
#endif

namespace morphology_operations {
/**
 * @brief dilate function. Performs dilation to the passed image
 *
 * @param image the input image
 * @return std::vector<std::vector<int32_t> > the resulted(dilated) image
 */
inline std::vector<std::vector<int32_t>> dilate(const std::vector<std::vector<int32_t>>& image) {
    Image img(image);
    assert(img.binary());
    int height = img._height();
    int width = img._width();
    Image resulted_img(height, width);
    for (int x = 1; x < height - 1; x++) {
        for (int y = 1; y < width - 1; y++) {
            if (!img.get_point(x - 1, y) || !img.get_point(x + 1, y) || !img.get_point(x, y - 1) ||
                !img.get_point(x, y + 1)) {
                resulted_img.set_point(x, y, 255);
            } else {
                img.set_point(x, y, image[x][y]);
            }
        }
    }
    return resulted_img.get_2d_array();
}

/**
 * @brief erote function. Performs erotion to the passed image
 *
 * @param image the input image
 * @return std::vector<std::vector<int32_t> > the resulted(eroted) image
 */
inline std::vector<std::vector<int32_t>> erote(const std::vector<std::vector<int32_t>>& image) {
    Image img(image);
    assert(img.binary());
    int height = img._height();
    int width = img._width();
    Image resulted_img(height, width);
    for (int x = 1; x < height - 1; x++) {
        for (int y = 1; y < width - 1; y++) {
            if (!img.get_point(x - 1, y) || !img.get_point(x + 1, y) ||
                !img.get_point(x, y - 1) && !img.get_point(x, y + 1)) {
                resulted_img.set_point(x, y, 0);
            } else {
                resulted_img.set_point(x, y, image[x][y]);
            }
        }
    }
    return resulted_img.get_2d_array();
}

/**
 * @brief open function. Performs opening to the passed image
 *
 * @param image the input image
 * @return std::vector<std::vector<int32_t> > the resulted(opened) image
 */
inline std::vector<std::vector<int32_t>> open(const std::vector<std::vector<int32_t>>& image) {
    Image img(image);
    assert(img.binary());
    std::vector<std::vector<int32_t>> resulted_img = erote(image); // first erote the image
    resulted_img = dilate(resulted_img);                           // then dilate the image
    return resulted_img;
}

/**
 * @brief close function. Performs closing to the passed image
 *
 * @param image the input image
 * @return std::vector<std::vector<int32_t> > the resulted(closed) image
 */
inline std::vector<std::vector<int32_t>> close(const std::vector<std::vector<int32_t>>& image) {
    Image img(image);
    assert(img.binary());
    std::vector<std::vector<int32_t>> resulted_img = dilate(image);
    resulted_img = erote(resulted_img);
    return resulted_img;
}
} // namespace morphology_operations

#endif
