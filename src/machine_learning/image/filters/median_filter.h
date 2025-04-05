#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H

#ifdef __cplusplus
#include <algorithm>
#include <iostream>
#include <vector>
#include "../image.h"
#endif

/**
 * @brief median_filter namespace
 */
namespace median_filter {
/**
 * @brief set_values function
 * @param values: 1d vector that holds the values of the mask
 * @param mat: the 2d (3x3)mask
 */
inline void set_values(std::vector<int32_t>& values, const std::vector<std::vector<int32_t>>& mat) {
    int height = mat.size(), width = mat[0].size();
    int index = 0;
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            values[index++] = mat[x][y];
        }
    }
}

/**
 * @brief apply_median_filter function: applies a 3x3 median filter on image img
 * @param img(Class Image): the input Image
 * @return vector<vector<int32_t> >: the resulted image after applying the 3x3
 * filter
 *
 */
inline std::vector<std::vector<int32_t>>
apply_median_filter(const std::vector<std::vector<int32_t>>& image) {
    Image img(image);
    int height = img._height();
    int width = img._width();
    Image resulted_img(height, width);
    std::vector<std::vector<int32_t>> mask = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<int32_t> values(9, 0);
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            mask[1][1] = img.get_point(x, y);
            if (x - 1 >= 0 && y - 1 >= 0) {
                mask[0][0] = img.get_point(x, y);
            }
            if (x - 1 >= 0) {
                mask[1][0] = img.get_point(x - 1, y);
            }
            if (y - 1 >= 0) {
                mask[0][1] = img.get_point(x, y - 1);
            }
            if (x + 1 < height && y - 1 >= 0) {
                mask[2][0] = img.get_point(x + 1, y - 1);
            }
            if (x - 1 >= 0 && y + 1 < width) {
                mask[0][2] = img.get_point(x - 1, y + 1);
            }
            if (y + 1 < width) {
                mask[1][2] = img.get_point(x, y + 1);
            }
            if (x + 1 < height && y + 1 < width) {
                mask[2][2] = img.get_point(x + 1, y + 1);
            }
            if (x + 1 < height) {
                mask[2][1] = img.get_point(x + 1, y);
            }
            set_values(values, mask);
            std::sort(values.begin(), values.end());
            resulted_img.set_point(x, y, values[5]);
        }
    }
    return resulted_img.get_2d_array();
}
} // namespace median_filter

#endif
