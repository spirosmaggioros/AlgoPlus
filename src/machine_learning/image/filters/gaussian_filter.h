#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#ifdef __cplusplus
#include <iostream>
#include "../image.h"
#endif

/**
 * @brief gaussian filter namespace
 */
namespace gaussian_filter {

/**
 * @brief apply_gaussian_filter function: applies a 3x3 gaussian filter to
 * passed image: image
 * @param image(Class Image): the input image
 * @return vector<vector<int32_t> >: the resulted image after applying the
 * gaussian filter
 */
inline std::vector<std::vector<int32_t>>
apply_gaussian_filter(const std::vector<std::vector<int32_t>>& image) {
    Image img(image);
    std::vector<std::vector<float>> kernel = {{1.0 / 16, 2.0 / 16, 1.0 / 16},
                                              {2.0 / 16, 4.0 / 16, 2.0 / 16},
                                              {1.0 / 16, 2.0 / 16, 1.0 / 16}};
    return img.apply_filter2d(kernel).get_2d_array();
}
} // namespace gaussian_filter

#endif
