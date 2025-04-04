#include "stddef.h"
#include "stdlib.h"

struct _Node {
    struct _Node* parent;
    struct _Node* left_child;
    struct _Node* right_child;

    double avg_value, threshold, loss;

    size_t nb_samples, depth, feature_idx;
};

// def new_node(depth: int) -> Node
struct _Node* new_node(size_t depth) {
    struct _Node* ret = malloc(sizeof(struct _Node));
    ret->left_child = ret->right_child = ret->parent = NULL;
    ret->depth = depth;
    return ret;
}
void clear_node(struct _Node* root) {
    if(root->left_child != NULL)
        clear_node(root->left_child);
    if(root->right_child != NULL)
        clear_node(root->right_child);
    free(root);
}
void _set_ys(struct _Node* node, double avg, double loss, size_t size) {
    node->avg_value = avg;
    node->loss = loss;
    node->nb_samples = size;
}
void _set_left_child(struct _Node* root, struct _Node* child) {
    root->left_child = child;
    child->parent = root;
}
void _set_right_child(struct _Node* root, struct _Node* child) {
    root->right_child = child;
    child->parent = root;
}
_Bool _is_root(struct _Node* root) {
    return root->parent == NULL;
}
_Bool _is_leaf(struct _Node* root) {
    return root->left_child == NULL || root->right_child == NULL;
}
