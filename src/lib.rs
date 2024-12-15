extern crate alloc;

use std::cmp::Ordering;

use alloc::vec::Vec;

/*
store the empty vector cells in a linked list. store head of linked list in Rosewood structure, and then
- if you need a new cell, use the head of the linked list, new head is head = storage[head].parent
- if you free a new cell, set storage[cell].parent = head, and then head = cell

store color information in the parent key ? reduces number of usable positions, but that should be fine in most
scenarios (with 64 bit usize, we can use 2^63 - 1 different keys).
other options: bitmap, bool in every node,
*/

#[derive(Debug, Default)]
#[repr(u8)]
enum NodeColor {
    #[default]
    Red,
    Black,
}

#[derive(Debug)]
struct RosewoodNode<K> {
    key: K,
    color: NodeColor,
    parent: usize,
    left: usize,
    right: usize,
}

impl<K> RosewoodNode<K> {
    fn new_isolated(key: K) -> Self {
        Self {
            key,
            color: NodeColor::default(),
            parent: 0,
            left: 0,
            right: 0,
        }
    }
}

impl<K: Default> Default for RosewoodNode<K> {
    fn default() -> Self {
        Self {
            key: K::default(),
            color: NodeColor::default(),
            parent: 0,
            left: 0,
            right: 0,
        }
    }
}

#[derive(Debug)]
pub struct Rosewood<K: PartialEq + Ord> {
    storage: Vec<RosewoodNode<K>>,
    root: usize,
}

impl<K: PartialEq + Ord> Rosewood<K> {
    const BLACK_NIL: usize = 0;

    pub fn contains(&self, key: K) -> bool {
        let mut current_node = self.root;

        while current_node != Self::BLACK_NIL {
            let curr_node_storage = &self.storage[current_node];

            match key.cmp(&curr_node_storage.key) {
                Ordering::Less => {
                    current_node = curr_node_storage.left;
                }
                Ordering::Equal => {
                    return true;
                }
                Ordering::Greater => {
                    current_node = curr_node_storage.right;
                }
            }
        }

        false
    }

    pub fn insert(&mut self, key: K) -> usize {
        let mut current_node = self.root;
        let mut parent_node = Self::BLACK_NIL;

        while current_node != Self::BLACK_NIL {
            parent_node = current_node;
            let curr_node_storage = &self.storage[current_node];

            if key < curr_node_storage.key {
                current_node = curr_node_storage.left;
            } else {
                current_node = curr_node_storage.right;
            }
        }

        let new_node_pos = self.storage.len();
        let parent_node_storage = &mut self.storage[parent_node];

        if parent_node == Self::BLACK_NIL {
            self.root = new_node_pos;
        } else if key < parent_node_storage.key {
            parent_node_storage.left = new_node_pos;
        } else {
            parent_node_storage.right = new_node_pos;
        }

        self.storage.push(RosewoodNode::new_isolated(key));
        self.storage[new_node_pos].parent = parent_node;

        self.fix_red_violation(new_node_pos);

        return parent_node;
    }

    fn fix_red_violation(&mut self, start_node_idx: usize) {
        let mut curr_node = start_node_idx;
        while matches!(
            self.storage[self.storage[curr_node].parent].color,
            NodeColor::Red
        ) {
            let parent_idx = self.storage[curr_node].parent;
            let grandparent_idx = self.storage[self.storage[curr_node].parent].parent;
            let grandparent = &self.storage[grandparent_idx];

            if grandparent_idx == Self::BLACK_NIL {
                self.storage[parent_idx].color = NodeColor::Black;
                return;
            }

            let parent_is_right_child = grandparent.right == parent_idx;
            let uncle = if parent_is_right_child {
                grandparent.left
            } else {
                grandparent.right
            };

            if matches!(self.storage[uncle].color, NodeColor::Red) {
                self.storage[parent_idx].color = NodeColor::Black;
                self.storage[uncle].color = NodeColor::Black;
                self.storage[grandparent_idx].color = NodeColor::Red;

                curr_node = grandparent_idx;
                continue;
            }

            let parent = &self.storage[parent_idx];
            if (parent_is_right_child && parent.left == curr_node)
                || (!parent_is_right_child && parent.right == curr_node)
            {
                if parent_is_right_child {
                    self.rotate_right(parent_idx);
                } else {
                    self.rotate_left(parent_idx);
                }

                curr_node = parent_idx;
                continue;
            }

            self.storage[parent_idx].color = NodeColor::Black;
            self.storage[grandparent_idx].color = NodeColor::Red;

            if parent_is_right_child {
                self.rotate_left(grandparent_idx);
            } else {
                self.rotate_right(grandparent_idx);
            }
        }
    }

    fn rotate_left(&mut self, center: usize) {
        let grandparent_idx = self.storage[center].parent;
        let sibling_idx = self.storage[center].right;

        let c_idx = self.storage[sibling_idx].left;

        self.storage[center].right = c_idx;
        self.storage[c_idx].parent = center;

        self.storage[sibling_idx].left = center;
        self.storage[center].parent = sibling_idx;
        self.storage[sibling_idx].parent = grandparent_idx;

        if grandparent_idx != Self::BLACK_NIL {
            if self.storage[grandparent_idx].right == center {
                self.storage[grandparent_idx].right = sibling_idx;
            } else {
                self.storage[grandparent_idx].left = sibling_idx;
            }
        } else {
            self.root = sibling_idx;
        }
    }

    fn rotate_right(&mut self, center: usize) {
        let grandparent_idx = self.storage[center].parent;
        let sibling_idx = self.storage[center].left;

        let c_idx = self.storage[sibling_idx].right;

        self.storage[center].left = c_idx;
        self.storage[c_idx].parent = center;

        self.storage[sibling_idx].right = center;
        self.storage[center].parent = sibling_idx;
        self.storage[sibling_idx].parent = grandparent_idx;

        if grandparent_idx != Self::BLACK_NIL {
            if self.storage[grandparent_idx].right == center {
                self.storage[grandparent_idx].right = sibling_idx;
            } else {
                self.storage[grandparent_idx].left = sibling_idx;
            }
        } else {
            self.root = sibling_idx;
        }
    }
}

impl<K: Default + PartialEq + Ord> Rosewood<K> {
    pub fn new() -> Self {
        Self {
            storage: alloc::vec![RosewoodNode::default()],
            root: Self::BLACK_NIL,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Rosewood;

    #[test]
    pub fn create_tree() {
        let tree = Rosewood::<usize>::new();
    }

    #[test]
    pub fn empty_tree_insertion() {
        let mut tree = Rosewood::<usize>::new();
        assert_eq!(tree.insert(5), 0);
        assert_eq!(tree.insert(7), 1);
        assert_eq!(tree.insert(9), 2);
        assert_eq!(tree.insert(3), 1);
    }
}
