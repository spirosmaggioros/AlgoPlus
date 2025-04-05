#ifndef SOBEL_OPERATOR_H
#define SOBEL_OPERATOR_H

#ifdef __cplusplus
#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
#include "../image.h"
#endif

/**
 * @brief Sobel operator is used for edge detection. It creates 2 3x3 arrays G_x
 * and G_y and apply them to an image A(G_x * A, G_y * A). After that, the
 * resulting image with edge detection is G = sqrt(G_x^2 + G_y^2)
 */
namespace sobel {
/**
 * @ brief kernel function. Of course you can change the kernel values depending
 * on your needs
 * @return pair<vector<int32_t>, vector<int32_t>> the G_x and G_y 3x3 kernels
 */
inline std::pair<std::vector<std::vector<int32_t>>, std::vector<std::vector<int32_t>>> kernel() {
    std::vector<std::vector<int32_t>> G_x = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    std::vector<std::vector<int32_t>> G_y = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    return std::make_pair(G_x, G_y);
}

/**
 * @brief square function
 * @param G the input 2d array
 * @return vector<vector<int32_t> > the squared root 2d array
 */
inline std::vector<std::vector<int32_t>> square(const std::vector<std::vector<int32_t>>& G) {
    int height = G.size();
    int width = G[0].size();
    std::vector<std::vector<int32_t>> res_img(height, std::vector<int32_t>(width));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            res_img[i][j] = sqrt(G[i][j]);
        }
    }
    return res_img;
}

/**
 * @brief Sobel apply function
 * @param image: the 2d array of the input image
 * @return vector<vector<in t32_t> > the resulting image
 */
inline std::vector<std::vector<int32_t>> Sobel(const std::vector<std::vector<int32_t>>& image) {
    Image img(image);

    std::pair<std::vector<std::vector<int32_t>>, std::vector<std::vector<int32_t>>> kernels =
        kernel();
    std::vector<std::vector<int32_t>> k1 = kernels.first;
    std::vector<std::vector<int32_t>> k2 = kernels.second;

    Image G_x(img.apply_filter2d(k1));
    Image G_y(img.apply_filter2d(k2));

    G_x = G_x.mul(G_x);
    G_y = G_y.mul(G_y);

    Image G(G_x);   // G = G_x^2
    G = G.add(G_y); // G += G_y^2

    return square(G.get_2d_array()); // result is: G = sqrt(G_x^2 + G_y^2)
}
} // namespace sobel

#endif
