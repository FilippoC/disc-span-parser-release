#pragma once

#include <memory>
#include <algorithm>
#include "disc_pstruct/tree.h"


// empty node at the right
void binarize(std::shared_ptr<Node> node, bool right=false);
void binarize_right(std::shared_ptr<Node> node);
void binarize_left(std::shared_ptr<Node> node);
void binarize(Tree& tree, bool right=false);
void binarize_right(Tree& tree);
void binarize_left(Tree& tree);

void unbinarize(std::shared_ptr<Node> node);
void unbinarize(Tree& tree);
