#ifndef STRING_HASHING_H
#define STRING_HASHING_H

#ifdef __cplusplus
#include <functional>
#include <string>
#include <unordered_map>
#endif

/**
 * @brief string hasher class
 * @details very useful for fast insert/search for strings as it
 * uses the builtin hash function
 */
class string_hasher {
  private:
    std::unordered_map<size_t, int> hash_table;

  public:
    explicit string_hasher() noexcept {}

    /**
     * @brief insert function
     * @param str the passed string
     */
    void insert(const std::string str) noexcept {
        size_t hashed = std::hash<std::string>{}(str);
        hash_table[hashed] = 1;
    }

    /**
     * @brief search function
     * @param str the passed string
     * @return true if str exists in the hasher
     * @return false otherwise
     */
    bool search(const std::string str) noexcept {
        size_t hashed = std::hash<std::string>{}(str);
        return hash_table[hashed] != 0;
    }

    /**
     * @brief size function
     * @return the size of the hasher(how many unique strings it contains)
     */
    unsigned long long size() { return hash_table.size(); }
};

#endif
