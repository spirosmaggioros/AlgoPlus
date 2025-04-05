#ifndef img_H
#define img_H

#ifdef __cplusplus
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#endif

class Image {
  private:
    std::vector<std::vector<int32_t>> img;
    int height;
    int width;

  public:
    /**
     * @brief image default constructor
     * @param img_pass: passed image(default empty)
     */
    explicit Image(std::vector<std::vector<int32_t>> img_pass = {}) {
        if (img_pass.empty()) {
            throw std::invalid_argument("Image dimensions can't be 0");
        }
        assert(!img_pass.empty());
        height = img_pass.size();
        width = img_pass[0].size();
        img = img_pass;
    }

    /**
     * @brief Image secondary constructor
     * @param height: the input height
     * @param width: the input width
     */
    explicit Image(int height, int width) {
        assert(height > 0);
        assert(width > 0);
        this->height = height;
        this->width = width;
        img = std::vector<std::vector<int32_t>>(height, std::vector<int32_t>(width, 0));
    }

    /**
     * @brief _height function
     * @return int: the height of the image
     */
    inline int _height() const { return this->height; }

    /**
     * @brief width function
     * @return int: the width of the image
     */
    inline int _width() const { return this->width; }

    /**
     * @brief get_2d_array function
     * @return vector<vector<int32_t> >: the 2d array of the image
     */
    inline std::vector<std::vector<int32_t>> get_2d_array() const { return this->img; }

    /**
     * @brief get_point function
     * @param x: the first dimension
     * @param y: the second dimension
     * @return int: the value of Img(x, y)
     */
    inline int get_point(const int x, const int y) const { return this->img[x][y]; }

    /**
     * @brief set_point function
     * @param x: the first dimension
     * @param y: the second dimension
     */
    inline void set_point(int x, int y, int val) { this->img[x][y] = val; }

    /**
     * @brief add_2_point function
     * @param x: first coordinate
     * @param y: second coordinate
     * @param val: the value you want to add to the point img(x, y)
     */
    template <typename T> inline void add_2_point(const int x, const int y, const T val) {
        img[x][y] += val;
    }

    /**
     * @brief get_point function
     * @param x: first coordinate
     * @param y: second coordinate
     * @return int32_t: the value of img(x, y)
     */
    inline int32_t get_point(const int x, const int y) { return img[x][y]; }

    /**
     * @brief binary function. Check if an image is black and white
     *
     * @return true if the image is black and white
     * @return false otherwise
     */
    inline bool binary() const {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (img[i][j] != 0 && img[i][j] != 255) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief add function
     * adds the img2 to img
     * @param img2: the image you want to add to the img
     * @return vector<vector<T> > the resulted image
     */
    template <typename T> inline Image add(const T img2) const {
        if constexpr (std::is_same_v<T, std::vector<std::vector<int32_t>>>) {
            assert(!img2.empty());
            assert(img2.size() == img.size());
            assert(img2[0].size() == img[0].size());
        } else {
            assert(img2._height() == img.size());
            assert(img2._width() == img[0].size());
        }

        Image resulted_img(height, width);
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                if constexpr (std::is_same_v<T, Image>) {
                    resulted_img.add_2_point(x, y, img[x][y] + img2.get_point(x, y));
                } else {
                    resulted_img.add_2_point(x, y, img[x][y] + img2[x][y]);
                }
            }
        }
        return resulted_img;
    }

    /**
     * @brief sub function
     * adds the img2 to img
     * @param img2: the image you want to subtract from the img
     * @return vector<vector<int32_t> > the resulted image
     */
    template <typename T> inline Image sub(const T img2) const {
        if constexpr (std::is_same_v<T, std::vector<std::vector<int32_t>>>) {
            assert(!img2.empty());
            assert(img2.size() == img.size());
            assert(img2[0].size() == img[0].size());
        } else {
            assert(img2._height() == img.size());
            assert(img2._width() == img[0].size());
        }

        Image resulted_img(height, width);
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                if constexpr (std::is_same_v<T, Image>) {
                    resulted_img.add_2_point(x, y, img[x][y] - img2.get_point(x, y));
                } else {
                    resulted_img.add_2_point(x, y, img[x][y] - img2[x][y]);
                }
            }
        }
        return resulted_img;
    }

    /**
     * @brief mul function
     * multiplies the img2 to img
     * @param img2: the image you want to subtract from the img
     * @return vector<vector<T> > the resulted image
     */
    template <typename T> inline Image mul(const T img2) const {
        if constexpr (std::is_same_v<T, std::vector<std::vector<int32_t>>>) {
            assert(!img2.empty());
            assert(img2.size() == img.size());
            assert(img2[0].size() == img[0].size());
        } else {
            assert(img2._height() == img.size());
            assert(img2._width() == img[0].size());
        }

        Image resulted_img(height, width);
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                if constexpr (std::is_same_v<T, Image>) {
                    resulted_img.add_2_point(x, y, img[x][y] * img2.get_point(x, y));
                } else {
                    resulted_img.add_2_point(x, y, img[x][y] * img2[x][y]);
                }
            }
        }
        return resulted_img;
    }

    /**
     * @brief apply_filter2d function
     * @param filter: 3x3 kernel to be applied to the image
     * @return vector<vector<T> > the resulted image
     */
    template <typename T> inline Image apply_filter2d(std::vector<std::vector<T>>& filter) const {
        assert(this->height > 0 && this->width > 0);
        assert(filter.size() == 3 && filter[0].size() == 3);

        Image resulted_img(height, width);
        int offsets[3][3][2] = {
            {{-1, -1}, {-1, 0}, {-1, 1}}, {{0, -1}, {0, 0}, {0, 1}}, {{1, -1}, {1, 0}, {1, 1}}};

        for (int x = 0; x < height; ++x) {
            for (int y = 0; y < width; ++y) {
                T value = 0;
                for (int fx = 0; fx < 3; ++fx) {
                    for (int fy = 0; fy < 3; ++fy) {
                        int nx = x + offsets[fx][fy][0];
                        int ny = y + offsets[fx][fy][1];

                        if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
                            value += img[nx][ny] * filter[fx][fy];
                        }
                    }
                }

                if constexpr (std::is_same_v<T, int32_t>) {
                    resulted_img.set_point(x, y, value);
                } else {
                    resulted_img.set_point(x, y, static_cast<int32_t>(round(value)));
                }
            }
        }
        return resulted_img;
    }

    /**
     * @brief overloaded operator << for Image class
     */
    inline friend std::ostream& operator<<(std::ostream& out, const Image& img) {
        int height = img._height(), width = img._width();

        auto write = [&](const int& start_at, const int& end_at) -> void {
            for (int i = start_at; i <= end_at && i < height; i++) {
                int j = 0;
                for (; j <= 10 && j < width; j++) {
                    if (j != std::min(10, width - 1)) {
                        out << img.get_point(i, j) << ", ";
                    } else {
                        out << img.get_point(i, j);
                    }
                }
                out << '\n';
            }
        };

        out << '[';
        write(0, 2);
        out << "...";
        out << '\n';
        write(height - 3, height - 1);
        out << "(height: " << height << ", width: " << width << ')' << ']';
        out << '\n';
        return out;
    }
};
#endif
