#pragma once

// For dependencies
inline
float& torch_get_unlabeled_dep_weight(float* const weights, const unsigned size, const int head, const int modifier)
{
    return weights[(head + 1) * (size + 1) + modifier + 1];
}

inline
const float& torch_get_unlabeled_dep_weight(const float* const weights, const unsigned size, const int head, const int modifier)
{
    return weights[(head + 1) * (size + 1) + modifier + 1];
}

// For constituents using a simple MLP weightning network (similar to the dependency one)
inline
float& torch_get_unlabeled_cst_weight(float* const weights, const unsigned size, const int i, const int j)
{
    return weights[i * size + j];
}

inline
const float& torch_get_unlabeled_cst_weight(const float* const weights, const unsigned size, const int i, const int j)
{
    return weights[i * size + j];
}

inline
float& torch_get_labeled_cst_weight(float* const weights, const unsigned size, const unsigned n_labels, const int i, const int j, const int label)
{
    return weights[i * size * n_labels + j * n_labels + label];
}

inline
const float& torch_get_labeled_cst_weight(const float* const weights, const unsigned size, const unsigned n_labels, const int i, const int j, const int label)
{
    return weights[i * size * n_labels + j * n_labels + label];
}

// For constituents using the fencepost model
// x = j
// y = i - 1
// but we do not take into account the BOS symbol in our parser,
// that is i=0 is the first word, not the BOS tag.
// therefore in the model we would have something like:
// i = i + 1 i.e. add one for the BOS tag
// j = j + 1
// x = j
// y = i - 1
// which simplifies to:
// x = j+1
// y = i
inline
float& torch_get_unlabeled_cst_weight_fencepost(float* const weights, const unsigned size, const int i, const int j)
{
    return weights[(j+1) * (size+1) + i];
}

inline
const float& torch_get_unlabeled_cst_weight_fencepost(const float* const weights, const unsigned size, const int i, const int j)
{
    return weights[(j+1) * (size+1) + i];
}

inline
float& torch_get_labeled_cst_weight_fencepost(float* const weights, const unsigned size, const unsigned n_labels, const int i, const int j, const int label)
{
    return weights[(j+1) * (size+1) * n_labels + i * n_labels + label];
}

inline
const float& torch_get_labeled_cst_weight_fencepost(const float* const weights, const unsigned size, const unsigned n_labels, const int i, const int j, const int label)
{
    return weights[(j+1) * (size+1) * n_labels + i * n_labels + label];
}
