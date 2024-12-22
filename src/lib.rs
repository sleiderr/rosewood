#![no_std]
extern crate alloc;

mod map;

pub mod containers {
    pub use crate::map::RosewoodMap;
}

use core::{
    cmp::{Ordering, max, min},
    mem::{swap, take},
};

use alloc::vec::Vec;

/*
store color information in the parent key ? reduces number of usable positions, but that should be fine in most
scenarios (with 64 bit usize, we can use 2^63 - 1 different keys).
other options: bitmap, bool in every node,
*/

#[derive(Clone, Copy, Debug, Default)]
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
enum RosewoodDeletionStep {
    Starting,
    Continue,
    ParentBlackSiblingBlackChildrenBlack,
    RedSibling,
    ParentRedChildrenBlack,
    CloseRedDistantBlack,
    DistantRed,
    Ended,
}

#[derive(Debug)]
pub struct Rosewood<K: PartialEq + Ord> {
    storage: Vec<RosewoodNode<K>>,
    free_nodes_head: usize,
    length: usize,
    root: usize,
}

impl<K: PartialEq + Ord> Rosewood<K> {
    const BLACK_NIL: usize = 0;

    pub fn contains(&self, key: &K) -> bool {
        return self.lookup(key) != Self::BLACK_NIL;
    }

    pub fn find_lower_bound(&self, target: &K) -> Option<&K> {
        self.lower_bound(target).map(|idx| &self.storage[idx].key)
    }

    pub fn find_lower_bound_mut(&mut self, target: &K) -> Option<&mut K> {
        self.lower_bound(target)
            .map(|idx| &mut self.storage[idx].key)
    }

    pub fn insert(&mut self, key: K) -> bool {
        let mut current_node = self.root;
        let mut parent_node = Self::BLACK_NIL;

        while current_node != Self::BLACK_NIL {
            parent_node = current_node;
            let curr_node_storage = &self.storage[current_node];

            if key < curr_node_storage.key {
                current_node = curr_node_storage.left;
            } else if key == curr_node_storage.key {
                return false;
            } else {
                current_node = curr_node_storage.right;
            }
        }

        let new_node_pos = self.find_free_slot_and_fill(key);

        if parent_node == Self::BLACK_NIL {
            self.root = new_node_pos;
        } else if self.storage[new_node_pos].key < self.storage[parent_node].key {
            self.storage[parent_node].left = new_node_pos;
        } else {
            self.storage[parent_node].right = new_node_pos;
        }

        self.storage[new_node_pos].parent = parent_node;
        self.storage[new_node_pos].left = Self::BLACK_NIL;
        self.storage[new_node_pos].right = Self::BLACK_NIL;

        self.fix_red_violation(new_node_pos);

        self.length += 1;
        true
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn max(&self) -> Option<&K> {
        let mut curr_elem = self.root;

        if curr_elem == Self::BLACK_NIL {
            return None;
        }

        loop {
            if self.storage[curr_elem].right == Self::BLACK_NIL {
                return Some(&self.storage[curr_elem].key);
            }

            curr_elem = self.storage[curr_elem].right;
        }
    }

    pub fn min(&self) -> Option<&K> {
        let mut curr_elem = self.root;

        if curr_elem == Self::BLACK_NIL {
            return None;
        }

        loop {
            if self.storage[curr_elem].left == Self::BLACK_NIL {
                return Some(&self.storage[curr_elem].key);
            }

            curr_elem = self.storage[curr_elem].left;
        }
    }

    pub fn remove(&mut self, key: &K) -> bool {
        match self.lookup(key) {
            Self::BLACK_NIL => false,
            idx => {
                self.delete(idx);

                true
            }
        }
    }

    fn lower_bound(&self, target: &K) -> Option<usize> {
        let mut walker = self.root;
        let mut best_fit_idx: Option<usize> = None;

        while walker != Self::BLACK_NIL {
            let walker_key = &self.storage[walker].key;

            match walker_key.cmp(&target) {
                Ordering::Less => {
                    walker = self.storage[walker].right;
                }
                Ordering::Equal => {
                    return Some(walker);
                }
                Ordering::Greater => {
                    best_fit_idx = match best_fit_idx {
                        Some(idx) => {
                            if let Ordering::Less = walker_key.cmp(&self.storage[idx].key) {
                                Some(walker)
                            } else {
                                Some(idx)
                            }
                        }
                        None => Some(walker),
                    };
                    walker = self.storage[walker].left;
                }
            }
        }

        best_fit_idx
    }

    #[inline]
    fn insert_free_slot(&mut self, slot: usize) {
        self.storage[slot].parent = self.free_nodes_head;

        self.free_nodes_head = slot;
    }

    #[inline]
    fn find_free_slot_and_fill(&mut self, key: K) -> usize {
        match self.free_nodes_head {
            Self::BLACK_NIL => {
                self.storage.push(RosewoodNode::new_isolated(key));

                self.storage.len() - 1
            }
            _ => {
                let free_slot = self.free_nodes_head;

                self.storage[free_slot].key = key;
                self.free_nodes_head = self.storage[free_slot].parent;

                free_slot
            }
        }
    }

    fn lookup(&self, key: &K) -> usize {
        let mut current_node = self.root;

        while current_node != Self::BLACK_NIL {
            let curr_node_storage = &self.storage[current_node];

            match key.cmp(&curr_node_storage.key) {
                Ordering::Less => {
                    current_node = curr_node_storage.left;
                }
                Ordering::Equal => {
                    return current_node;
                }
                Ordering::Greater => {
                    current_node = curr_node_storage.right;
                }
            }
        }

        Self::BLACK_NIL
    }

    fn delete(&mut self, node_idx: usize) {
        self.length -= 1;

        match (self.storage[node_idx].left, self.storage[node_idx].right) {
            (Self::BLACK_NIL, Self::BLACK_NIL) => {
                if self.root == node_idx {
                    self.root = Self::BLACK_NIL;
                    self.insert_free_slot(node_idx);
                } else {
                    if matches!(self.storage[node_idx].color, NodeColor::Red) {
                        let parent_idx = self.storage[node_idx].parent;
                        let parent = &mut self.storage[parent_idx];

                        if parent.left == node_idx {
                            parent.left = Self::BLACK_NIL;
                        } else {
                            parent.right = Self::BLACK_NIL;
                        }

                        self.insert_free_slot(node_idx);
                    } else {
                        self.delete_black_leaf(node_idx);
                    }
                }

                return;
            }
            (single_child_idx, Self::BLACK_NIL) | (Self::BLACK_NIL, single_child_idx) => {
                let parent_idx = self.storage[node_idx].parent;
                let parent = &mut self.storage[parent_idx];

                if parent.left == node_idx {
                    parent.left = single_child_idx;
                } else {
                    parent.right = single_child_idx;
                }

                if self.root == node_idx {
                    self.root = single_child_idx;
                }

                self.storage[single_child_idx].parent = parent_idx;

                self.insert_free_slot(node_idx);

                return;
            }
            (left_child_idx, _right_child_idx) => {
                let succ = self.find_inorder_predecessor(left_child_idx);
                self.swap_nodes(succ, node_idx);

                self.insert_free_slot(succ);
            }
        }
    }

    fn swap_nodes(&mut self, node_a: usize, node_b: usize) {
        if node_a == node_b {
            return;
        }

        let (part_a, part_b) = self.storage.split_at_mut(min(node_a, node_b) + 1);

        swap(
            &mut part_a[part_a.len() - 1].key,
            &mut part_b[max(node_a, node_b) - min(node_a, node_b) - 1].key,
        );
    }

    fn find_inorder_predecessor(&self, subtree_root: usize) -> usize {
        let mut curr_node = subtree_root;

        loop {
            let next_node = self.storage[curr_node].right;

            if next_node == Self::BLACK_NIL {
                return curr_node;
            }
            curr_node = next_node;
        }
    }

    fn delete_black_leaf(&mut self, node_idx: usize) {
        let mut step = RosewoodDeletionStep::Starting;
        let mut curr_node = node_idx;
        let mut parent_idx = self.storage[curr_node].parent;
        let mut is_right_child = self.storage[parent_idx].right == curr_node;
        let (mut sibling_idx, mut distant_nephew_idx, mut close_nephew_idx) =
            (Self::BLACK_NIL, Self::BLACK_NIL, Self::BLACK_NIL);

        loop {
            match step {
                RosewoodDeletionStep::Starting => {
                    if is_right_child {
                        self.storage[parent_idx].right = Self::BLACK_NIL;
                    } else {
                        self.storage[parent_idx].left = Self::BLACK_NIL;
                    }
                    step = RosewoodDeletionStep::Continue;
                }
                RosewoodDeletionStep::Continue => {
                    parent_idx = self.storage[curr_node].parent;
                    is_right_child = self.storage[parent_idx].right == curr_node;

                    if is_right_child {
                        self.storage[parent_idx].right = Self::BLACK_NIL;
                    } else {
                        self.storage[parent_idx].left = Self::BLACK_NIL;
                    }

                    (sibling_idx, distant_nephew_idx, close_nephew_idx) = if is_right_child {
                        let sibling_idx = self.storage[parent_idx].left;
                        (
                            sibling_idx,
                            self.storage[sibling_idx].left,
                            self.storage[sibling_idx].right,
                        )
                    } else {
                        let sibling_idx = self.storage[parent_idx].right;
                        (
                            sibling_idx,
                            self.storage[sibling_idx].right,
                            self.storage[sibling_idx].left,
                        )
                    };

                    if matches!(self.storage[sibling_idx].color, NodeColor::Red) {
                        step = RosewoodDeletionStep::RedSibling;
                        continue;
                    }

                    if distant_nephew_idx != Self::BLACK_NIL
                        && matches!(self.storage[distant_nephew_idx].color, NodeColor::Red)
                    {
                        step = RosewoodDeletionStep::DistantRed;
                        continue;
                    }

                    if close_nephew_idx != Self::BLACK_NIL
                        && matches!(self.storage[close_nephew_idx].color, NodeColor::Red)
                    {
                        step = RosewoodDeletionStep::CloseRedDistantBlack;
                        continue;
                    }

                    if matches!(self.storage[parent_idx].color, NodeColor::Red) {
                        step = RosewoodDeletionStep::ParentRedChildrenBlack;
                        continue;
                    }

                    step = RosewoodDeletionStep::ParentBlackSiblingBlackChildrenBlack;
                    continue;
                }
                RosewoodDeletionStep::ParentBlackSiblingBlackChildrenBlack => {
                    self.storage[sibling_idx].color = NodeColor::Red;
                    curr_node = parent_idx;

                    step = RosewoodDeletionStep::Continue;
                }
                RosewoodDeletionStep::RedSibling => {
                    if is_right_child {
                        self.rotate_right(parent_idx);
                    } else {
                        self.rotate_left(parent_idx);
                    }

                    self.storage[parent_idx].color = NodeColor::Red;
                    self.storage[sibling_idx].color = NodeColor::Black;

                    sibling_idx = close_nephew_idx;
                    distant_nephew_idx = if is_right_child {
                        self.storage[sibling_idx].left
                    } else {
                        self.storage[sibling_idx].right
                    };

                    step = RosewoodDeletionStep::ParentRedChildrenBlack;

                    if distant_nephew_idx != Self::BLACK_NIL
                        && matches!(self.storage[distant_nephew_idx].color, NodeColor::Red)
                    {
                        step = RosewoodDeletionStep::DistantRed;
                    }

                    close_nephew_idx = if is_right_child {
                        self.storage[sibling_idx].right
                    } else {
                        self.storage[sibling_idx].left
                    };

                    if close_nephew_idx != Self::BLACK_NIL
                        && matches!(self.storage[close_nephew_idx].color, NodeColor::Red)
                    {
                        step = RosewoodDeletionStep::CloseRedDistantBlack;
                    }

                    continue;
                }
                RosewoodDeletionStep::ParentRedChildrenBlack => {
                    self.storage[sibling_idx].color = NodeColor::Red;
                    self.storage[parent_idx].color = NodeColor::Black;

                    step = RosewoodDeletionStep::Ended;
                    continue;
                }
                RosewoodDeletionStep::CloseRedDistantBlack => {
                    if is_right_child {
                        self.rotate_left(sibling_idx);
                    } else {
                        self.rotate_right(sibling_idx);
                    }
                    self.storage[sibling_idx].color = NodeColor::Red;
                    self.storage[close_nephew_idx].color = NodeColor::Black;

                    step = RosewoodDeletionStep::DistantRed;
                    continue;
                }
                RosewoodDeletionStep::DistantRed => {
                    if is_right_child {
                        self.rotate_right(parent_idx);
                    } else {
                        self.rotate_left(parent_idx);
                    }

                    self.storage[sibling_idx].color = self.storage[parent_idx].color;
                    self.storage[parent_idx].color = NodeColor::Black;
                    self.storage[distant_nephew_idx].color = NodeColor::Black;

                    step = RosewoodDeletionStep::Ended;
                    continue;
                }
                RosewoodDeletionStep::Ended => {
                    return;
                }
            }
        }
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
            length: 0,
            free_nodes_head: 0,
            root: Self::BLACK_NIL,
        }
    }

    pub fn extract_lower_bound(&mut self, target: &K) -> Option<K> {
        let lower_bound = self.lower_bound(target)?;
        let key = take(&mut self.storage[lower_bound].key);
        self.delete(lower_bound);

        Some(key)
    }
}

#[cfg(test)]
mod tests {
    use crate::Rosewood;

    #[test]
    pub fn root_removal() {
        let mut tree = Rosewood::<usize>::new();

        tree.insert(5);
        tree.insert(6);
        tree.insert(2);
        tree.insert(19);
        tree.insert(12);
        tree.insert(4);

        tree.remove(&5);

        assert_eq!(tree.storage[tree.root].key, 4);
    }

    #[test]
    pub fn tree_min_max() {
        let mut tree = Rosewood::<usize>::new();

        tree.insert(5);
        tree.insert(4);
        tree.insert(3);
        tree.insert(3);

        assert_eq!(*tree.min().unwrap(), 3);

        tree.remove(&3);
        tree.remove(&5);

        assert_eq!(*tree.max().unwrap(), 4);
    }

    #[test]
    pub fn tree_length() {
        let mut tree = Rosewood::<usize>::new();

        tree.insert(5);
        tree.insert(4);
        tree.insert(3);
        tree.insert(3);

        assert_eq!(tree.len(), 3);

        tree.remove(&4);
        tree.remove(&7);

        assert_eq!(tree.len(), 2);
    }
}
