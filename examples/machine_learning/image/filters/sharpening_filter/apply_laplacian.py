import matplotlib.pyplot as plt
import json
import os


if __name__ == "__main__":
    os.system("c++ -std=c++17 main.cc")
    os.system("./a.out")

    fig = plt.figure(figsize=(10, 7))
    original_data = open("data.json")
    file = json.load(original_data)
    data = file['img']
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(data, cmap='gray')
    filename = open("results.json")
    file = json.load(filename)
    data = file['data']
    plt.subplot(1, 2, 2)
    plt.title("After applying Sharpening filter")
    plt.imshow(data, cmap='gray')
    plt.show()
