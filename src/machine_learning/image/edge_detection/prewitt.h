#ifndef PREWITT_H
#define PREWITT_H

#ifdef __cplusplus
#include <cmath>
#include <iostream>
#include <vector>
#include "../image.h"
#endif

/**
 * @brief Prewitt operator is used for edge detection. It creates 2 3x3 arrays
 * G_x and G_y and apply them to an image A(G_x * A, G_y * A). After that, the
 * resulting image with edge detection is G = sqrt(G_x^2 + G_y^2). It's almost
 * the same as the sobel operator, but with a different mask
 */
namespace prewitt {
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
 * @brief Prewitt apply function.
 *
 * @param image the input image
 * @return std::vector<std::vector<int32_t> > the resulted image after applying
 * the prewitt mask
 */
inline std::vector<std::vector<int32_t>> Prewitt(const std::vector<std::vector<int32_t>>& image) {
    Image img(image);
    std::vector<std::vector<int32_t>> k1 = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    std::vector<std::vector<int32_t>> k2 = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};

    Image G_x(img.apply_filter2d(k1));
    Image G_y(img.apply_filter2d(k2));

    G_x = G_x.mul(G_x);
    G_y = G_y.mul(G_y);

    Image G(G_x);   // G = G_x^2
    G = G.add(G_y); // G += G_y^2

    return square(G.get_2d_array()); // result is: G = sqrt(G_x^2 + G_y^2)
}
} // namespace prewitt

#endif
