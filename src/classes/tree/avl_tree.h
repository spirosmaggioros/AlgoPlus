#ifndef AVL_TREE_H
#define AVL_TREE_H

#ifdef ENABLE_TREE_VISUALIZATION
#include "../../visualization/tree_visual/tree_visualization.h"
#endif

#ifdef __cplusplus
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <vector>
#endif

/**
 *@brief Class for AVL tree.
 */
template <typename T> class avl_tree {
  public:
    /**
     *@brief Contructor for AVL tree class.
     *@param __elements: you can directly pass a vector<T> so you don't have to do
     *insert multiple times.
     */
    inline explicit avl_tree(std::vector<T> _elements = {}) noexcept : root(nullptr) {
        if (!_elements.empty()) {
            for (T& x : _elements) {
                this->insert(x);
            }
        }
    }

    /**
     * @brief Copy constructor for avl tree class
     * @param a the tree we want to copy
     */
    inline explicit avl_tree(const avl_tree& a) : root(a.root), _size(a._size) {}

    /**
     * @brief operator = for avl tree class
     * @param a the tree we want to copy
     * @return avl_tree&
     */
    inline avl_tree& operator=(const avl_tree& a) {
        root = a.root;
        _size = a._size;
        return *this;
    }

    /**
     * @brief Destroy the avl tree object
     *
     */
    inline ~avl_tree() noexcept {}

    /**
     *@brief insert function.
     *@param key: key to be inserted.
     */
    inline void insert(T key) {
        root = _insert(root, key);
        _size++;
    }

    /**
     *@brief clear function
     *Erase all the nodes from the tree.
     */
    inline void clear() {
        root = nullptr;
        _size = 0;
        return;
    }

    /**
     * @brief get_root function
     * @return T: the root value
     * Created for bubble.h container
     */
    inline T get_root() { return this->root->info; }

    /**
     *@brief search function.
     *@param key: key to be searched.
     *@returns true if the key exists in the tree.
     */
    inline bool search(T key) { return _search(root, key); }

    class Iterator;

    /**
     * @brief pointer that points to begin
     *
     * @return Iterator
     */
    inline Iterator begin() {
        std::vector<T> ino = this->inorder();
        return Iterator(0, ino);
    }

    /**
     * @brief pointer that points to end
     *
     * @return Iterator
     */
    inline Iterator end() {
        std::vector<T> ino = this->inorder();
        return Iterator(ino.size(), ino);
    }

    /**
     * @brief size function
     *
     * @return size_t the size of the tree
     */
    inline size_t size() const { return _size; }

    /**
     *@brief remove function.
     *@param key: key to be removed.
     */
    inline void remove(T key) { root = _remove(root, key); }

    /**
     *@brief inorder function.
     *@returns vector<T>, the elements inorder.
     */
    inline std::vector<T> inorder() const {
        std::vector<T> path;
        _inorder([&](std::shared_ptr<node> callbacked) { path.push_back(callbacked->info); }, root);
        return path;
    }
    /**
    @brief preorder function.
    *@returns vector<T>, the elements preorder.
    */
    inline std::vector<T> preorder() const {
        std::vector<T> path;
        _preorder([&](std::shared_ptr<node> callbacked) { path.push_back(callbacked->info); },
                  root);
        return path;
    }
    /**
     *@brief postorder function.
     *@returns vector<T>, the elements postorder.
     */
    inline std::vector<T> postorder() const {
        std::vector<T> path;
        _postorder([&](std::shared_ptr<node> callbacked) { path.push_back(callbacked->info); },
                   root);
        return path;
    }

    /**
     *@brief level order function.
     *@returns vector<T>, the level order traversal of the tree
     */
    inline std::vector<std::vector<T>> level_order() {
        std::vector<std::vector<T>> path;
        std::queue<std::shared_ptr<node>> q;
        q.push(root);
        while (!q.empty()) {
            size_t size = q.size();
            std::vector<T> level;
            for (size_t i = 0; i < size; i++) {
                std::shared_ptr<node> current = q.front();
                q.pop();
                level.push_back(current->info);
                if (current->left) {
                    q.push(current->left);
                }
                if (current->right) {
                    q.push(current->right);
                }
            }
            path.push_back(level);
        }
        return path;
    }

    /**
     *@brief visualize function
     *@returns .dot file that can be previewed using graphviz in vscode.
     */

#ifdef TREE_VISUALIZATION_H
    inline void visualize() {
        std::string _generated = generate_visualization();
        tree_visualization::visualize(_generated);
    }
#endif

    /**
     * @brief operator << for avl_tree class
     */
    inline friend std::ostream& operator<<(std::ostream& out, avl_tree<T>& t) {
        std::vector<T> order = t.inorder();
        for (int i = 0; i < order.size(); i++) {
            if (i != order.size() - 1) {
                out << order[i] << ", ";
            } else {
                out << order[i] << '\n';
            }
        }
        return out;
    }

  private:
    /**
     *@brief Struct for the node type pointer.
     *@param info: the value of the node.
     *@param height: height of each node.
     *@param left: pointer to the left.
     *@param right: pointer to the right.
     */
    typedef struct node {
        T info;
        int64_t height{0};
        std::shared_ptr<node> left;
        std::shared_ptr<node> right;
        node(T key) : info(key), left(nullptr), right(nullptr) {}
    } node;

    std::shared_ptr<node> root;
    size_t _size{};

    int64_t height(std::shared_ptr<node> root) {
        if (root == nullptr)
            return 0;
        return 1 + std::max(height(root->left), height(root->right));
    }

    std::shared_ptr<node> createNode(T info) {
        std::shared_ptr<node> nn = std::make_shared<node>(info);
        return nn;
    }

    int64_t getBalance(std::shared_ptr<node> root) {
        return height(root->left) - height(root->right);
    }

    std::shared_ptr<node> rightRotate(std::shared_ptr<node> root) {
        std::shared_ptr<node> t = root->left;
        std::shared_ptr<node> u = t->right;
        t->right = root;
        root->left = u;
        return t;
    }

    std::shared_ptr<node> leftRotate(std::shared_ptr<node> root) {
        std::shared_ptr<node> t = root->right;
        std::shared_ptr<node> u = t->left;
        t->left = root;
        root->right = u;
        return t;
    }

    std::shared_ptr<node> minValue(std::shared_ptr<node> root) {
        if (root->left == nullptr)
            return root;
        return minValue(root->left);
    }

    std::shared_ptr<node> _insert(std::shared_ptr<node> root, T item) {
        std::shared_ptr<node> nn = createNode(item);
        if (root == nullptr)
            return nn;
        if (item < root->info) {
            root->left = _insert(root->left, item);
        } else if (item > root->info) {
            root->right = _insert(root->right, item);
        } else {
            return root;
        }
        int b = getBalance(root);
        if (b > 1) {
            if (getBalance(root->left) < 0)
                root->left = leftRotate(root->left);
            return rightRotate(root);
        } else if (b < -1) {
            if (getBalance(root->right) > 0)
                root->right = rightRotate(root->right);
            return leftRotate(root);
        }
        return root;
    }

    std::shared_ptr<node> _remove(std::shared_ptr<node> root, T key) {
        if (root == nullptr)
            return root;
        if (key < root->info)
            root->left = _remove(root->left, key);
        else if (key > root->info)
            root->right = _remove(root->right, key);

        else {
            if (!root->right) {
                std::shared_ptr<node> temp = root->left;
                root = nullptr;
                _size--;
                return temp;
            } else if (!root->left) {
                std::shared_ptr<node> temp = root->right;
                root = nullptr;
                _size--;
                return temp;
            }
            std::shared_ptr<node> temp = minValue(root->right);
            root->info = temp->info;
            root->right = _remove(root->right, temp->info);
        }
        return root;
    }

    bool _search(std::shared_ptr<node> root, T key) {
        while (root) {
            if (root->info < key) {
                root = root->right;
            } else if (root->info > key) {
                root = root->left;
            } else {
                return true;
            }
        }
        return false;
    }

    void _inorder(std::function<void(std::shared_ptr<node>)> callback,
                  std::shared_ptr<node> root) const {
        if (root) {
            _inorder(callback, root->left);
            callback(root);
            _inorder(callback, root->right);
        }
    }

    void _postorder(std::function<void(std::shared_ptr<node>)> callback,
                    std::shared_ptr<node> root) const {
        if (root) {
            _inorder(callback, root->left);
            _inorder(callback, root->right);
            callback(root);
        }
    }

    void _preorder(std::function<void(std::shared_ptr<node>)> callback,
                   std::shared_ptr<node> root) const {
        if (root) {
            callback(root);
            _inorder(callback, root->left);
            _inorder(callback, root->right);
        }
    }

    std::string generate_visualization() {
        std::string _generate = _inorder_gen(root);
        return _generate;
    }

    std::string _inorder_gen(std::shared_ptr<node> root) {
        std::string _s;
        if (std::is_same_v<T, char> || std::is_same_v<T, std::string>) {
            if (root->left) {
                _s += root->info;
                _s += "->";
                _s += root->left->info;
                _s += "\n";
                _s += _inorder_gen(root->left);
            }
            if (root->right) {
                _s += root->info;
                _s += "->";
                _s += root->right->info;
                _s += "\n";
                _s += _inorder_gen(root->right);
            }
        } else {
            if (root->left) {
                _s += std::to_string(root->info) + "->" + std::to_string(root->left->info) + "\n" +
                      _inorder_gen(root->left);
            }
            if (root->right) {
                _s += std::to_string(root->info) + "->" + std::to_string(root->right->info) + "\n" +
                      _inorder_gen(root->right);
            }
        }
        return _s;
    }
};

/**
 * @brief Iterator class
 */
template <typename T> class avl_tree<T>::Iterator {
  private:
    std::vector<T> elements;
    int64_t index;

  public:
    /**
     * @brief Construct a new Iterator object
     *
     * @param els vector<T> - the elements in inorder fashion
     */
    explicit Iterator(const int64_t& index, std::vector<T>& els) noexcept
        : index(index), elements(els) {}

    /**
     * @brief = operator for Iterator type
     *
     * @param index the current index
     * @return Iterator&
     */
    Iterator& operator=(int64_t index) {
        this->index = index;
        return *(this);
    }

    /**
     * @brief operator ++ for type Iterator
     *
     * @return Iterator&
     */
    Iterator& operator++() {
        if (this->index < elements.size()) {
            this->index++;
        }
        return *(this);
    }

    /**
     * @brief operator ++ for type Iterator
     *
     * @return Iterator
     */
    Iterator operator++(int) {
        Iterator it = *this;
        ++*(this);
        return it;
    }

    /**
     * @brief operator -- for type Iterator
     *
     * @return Iterator&
     */
    Iterator& operator--() {
        if (this->index > 0) {
            this->index--;
        }
        return *(this);
    }

    /**
     * @brief operator -- for type Iterator
     *
     * @return Iterator
     */
    Iterator operator--(int) {
        Iterator it = *this;
        --*(this);
        return it;
    }

    /**
     * @brief operator != for type Iterator
     *
     * @param it const Iterator
     * @return true if the current element that exist in the index is not equal to
     * the it.element that exist in the it.index
     * @return false otherwise
     */
    bool operator!=(const Iterator& it) {
        return index != it.index && elements[index] != it.elements[it.index];
    }

    /**
     * @brief operator * for type Iterator
     *
     * @return T the value of the node
     */
    T operator*() { return elements[index]; }
};

#endif
