#ifndef LAPLACIAN_POINT_DETECT_H
#define LAPLACIAN_POINT_DETECT_H

#ifdef __cplusplus
#include <iostream>
#include <vector>
#include "../image.h"
#endif

namespace laplacian_detection {
/**
 * @brief point detection using a 3x3 laplacian mask
 *
 * @param img the input image
 * @param threshold  threshold(as threhold goes up, the model becomes more
 * selective)
 * @return std::vector<std::vector<int32_t> >
 */
inline std::vector<std::vector<int32_t>>
apply_point_detection(const std::vector<std::vector<int32_t>> img, const double threshold) {
    int height = img.size();
    int width = img[0].size();
    std::vector<std::vector<int32_t>> kernel = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
    Image resulted_img(img);
    for (int x = 1; x < height - 1; x++) {
        for (int y = 1; y < width - 1; y++) {
            int padding = img[x - 1][y - 1] + img[x][y - 1] + img[x + 1][y + 1] + img[x][y + 1] +
                          img[x + 1][y] + img[x + 1][y - 1] + img[x - 1][y];
            if (8 * img[x][y] - padding >= threshold) {
                resulted_img.add_2_point(x, y, img[x][y] * kernel[1][1]);
                if (y - 1 >= 0) {
                    resulted_img.add_2_point(x, y, img[x][y - 1] * kernel[1][0]);
                }
                if (y + 1 < width) {
                    resulted_img.add_2_point(x, y, img[x][y + 1] * kernel[1][2]);
                    if (x + 1 < height) {
                        resulted_img.add_2_point(x, y, img[x + 1][y + 1] * kernel[2][2]);
                    }
                }
                if (x - 1 >= 0) {
                    resulted_img.add_2_point(x, y, img[x - 1][y] * kernel[0][1]);
                    if (y + 1 < width) {
                        resulted_img.add_2_point(x, y, img[x - 1][y + 1] * kernel[0][2]);
                    }
                    if (y - 1 >= 0) {
                        resulted_img.add_2_point(x, y, img[x - 1][y - 1] * kernel[0][0]);
                    }
                }
                if (x + 1 < height) {
                    resulted_img.add_2_point(x, y, img[x + 1][y] * kernel[2][1]);
                    if (y - 1 >= 0) {
                        resulted_img.add_2_point(x, y, img[x + 1][y - 1] * kernel[2][0]);
                    }
                }
            } else {
                resulted_img.set_point(x, y, 255);
            }
        }
    }
    return resulted_img.get_2d_array();
}
} // namespace laplacian_detection

#endif
