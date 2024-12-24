extern crate alloc;

mod iter;
mod map;

pub mod iterators {
    pub use crate::iter::*;
}

pub mod containers {
    pub use crate::map::RosewoodMap;
}

use core::{
    cmp::{Ordering, max, min},
    marker::PhantomData,
    mem::{swap, take},
    ptr,
};

use alloc::vec::Vec;
use iter::{RosewoodSortedIterator, RosewoodSortedIteratorMut};

/*
store color information in the parent key ? reduces number of usable positions, but that should be fine in most
scenarios (with 64 bit usize, we can use 2^63 - 1 different keys).
other options: bitmap, bool in every node,
*/

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(u8)]
enum NodeColor {
    #[default]
    Red,
    Black,
    Free,
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
    UpdateVariables,
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

    pub fn capacity(&self) -> usize {
        self.storage.capacity()
    }

    pub fn contains(&self, key: &K) -> bool {
        self.lookup(key) != Self::BLACK_NIL
    }

    #[must_use]
    pub fn iter(&self) -> RosewoodSortedIterator<'_, K> {
        RosewoodSortedIterator {
            tree: self,
            curr: self.root,
            stack: alloc::vec![],
        }
    }

    #[must_use]
    pub fn iter_mut(&mut self) -> RosewoodSortedIteratorMut<'_, K> {
        RosewoodSortedIteratorMut {
            tree: ptr::from_mut(self),
            curr: self.root,
            stack: alloc::vec![],
            phantom: PhantomData {},
        }
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

            match key.cmp(&curr_node_storage.key) {
                Ordering::Less => {
                    current_node = curr_node_storage.left;
                }
                Ordering::Equal => {
                    return false;
                }
                Ordering::Greater => {
                    current_node = curr_node_storage.right;
                }
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
        self.storage[new_node_pos].color = NodeColor::Red;

        self.fix_red_violation(new_node_pos);

        self.length += 1;
        true
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    #[must_use]
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

    #[must_use]
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
        let removal_success = match self.lookup(key) {
            Self::BLACK_NIL => false,
            idx => {
                self.delete(idx);

                true
            }
        };

        if self.available_slots() > self.length {
            self.shrink();
        }

        removal_success
    }

    #[inline]
    #[must_use]
    fn available_slots(&self) -> usize {
        self.storage.len() - self.length - 1
    }

    fn shrink(&mut self) {
        let mid = (self.storage.len() >> 1) + 1;
        let (front, back) = self.storage.split_at_mut(mid);

        let mut front_idx = 1;

        for idx in 0..back.len() {
            if matches!(back[idx].color, NodeColor::Black | NodeColor::Red) {
                loop {
                    if matches!(front[front_idx].color, NodeColor::Free) {
                        swap(&mut front[front_idx], &mut back[idx]);
                        let parent_idx = front[front_idx].parent;
                        let left_idx = front[front_idx].left;
                        let right_idx = front[front_idx].right;

                        if self.root == mid + idx {
                            self.root = front_idx;
                        }

                        if parent_idx != Self::BLACK_NIL {
                            if parent_idx >= mid {
                                if back[parent_idx - mid].right == mid + idx {
                                    back[parent_idx - mid].right = front_idx
                                } else {
                                    back[parent_idx - mid].left = front_idx
                                }
                            } else {
                                if front[parent_idx].right == mid + idx {
                                    front[parent_idx].right = front_idx
                                } else {
                                    front[parent_idx].left = front_idx
                                }
                            }
                        }

                        if left_idx != Self::BLACK_NIL {
                            if left_idx >= mid {
                                back[left_idx - mid].parent = front_idx;
                            } else {
                                front[left_idx].parent = front_idx;
                            }
                        }

                        if right_idx != Self::BLACK_NIL {
                            if right_idx >= mid {
                                back[right_idx - mid].parent = front_idx;
                            } else {
                                front[right_idx].parent = front_idx;
                            }
                        }

                        break;
                    }

                    front_idx += 1;
                }
            }
        }

        self.storage.truncate(mid + 1);
        self.storage.shrink_to_fit();

        self.rebuild_free_nodes_list();
    }

    fn rebuild_free_nodes_list(&mut self) {
        self.free_nodes_head = 0;

        for (idx, slot) in self.storage.iter_mut().enumerate() {
            if matches!(slot.color, NodeColor::Free) {
                swap(&mut self.free_nodes_head, &mut slot.parent);
                self.free_nodes_head = idx;
            }
        }
    }

    fn lower_bound(&self, target: &K) -> Option<usize> {
        let mut walker = self.root;
        let mut best_fit_idx: Option<usize> = None;

        while walker != Self::BLACK_NIL {
            let walker_key = &self.storage[walker].key;

            match walker_key.cmp(target) {
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
        self.storage[slot].color = NodeColor::Free;

        self.free_nodes_head = slot;
    }

    #[inline]
    fn find_free_slot_and_fill(&mut self, key: K) -> usize {
        if self.free_nodes_head == Self::BLACK_NIL {
            self.storage.push(RosewoodNode::new_isolated(key));

            self.storage.len() - 1
        } else {
            let free_slot = self.free_nodes_head;

            self.storage[free_slot].key = key;
            self.free_nodes_head = self.storage[free_slot].parent;

            free_slot
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
                } else if matches!(self.storage[node_idx].color, NodeColor::Red) {
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
                    self.insert_free_slot(node_idx);
                }
            }
            (single_child_idx, Self::BLACK_NIL) | (Self::BLACK_NIL, single_child_idx) => {
                let parent_idx = self.storage[node_idx].parent;
                let parent = &mut self.storage[parent_idx];

                if parent_idx != Self::BLACK_NIL {
                    if parent.left == node_idx {
                        parent.left = single_child_idx;
                    } else {
                        parent.right = single_child_idx;
                    }
                }

                if self.root == node_idx {
                    self.root = single_child_idx;
                }

                self.storage[single_child_idx].parent = parent_idx;
                self.storage[single_child_idx].color = NodeColor::Black;

                self.insert_free_slot(node_idx);
            }
            (left_child_idx, _right_child_idx) => {
                self.length += 1;
                let succ = self.find_inorder_predecessor(left_child_idx);
                self.swap_nodes(succ, node_idx);

                self.delete(succ);
            }
        }

        debug_assert!(self.storage[Self::BLACK_NIL].color == NodeColor::Black);
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
                    step = RosewoodDeletionStep::UpdateVariables;
                }
                RosewoodDeletionStep::Continue => {
                    is_right_child = self.storage[parent_idx].right == curr_node;
                    step = RosewoodDeletionStep::UpdateVariables;
                }
                RosewoodDeletionStep::UpdateVariables => {
                    parent_idx = self.storage[curr_node].parent;

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

                    if parent_idx == Self::BLACK_NIL {
                        step = RosewoodDeletionStep::Ended;
                        continue;
                    }

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
                    if sibling_idx != Self::BLACK_NIL {
                        self.storage[sibling_idx].color = NodeColor::Red;
                    }
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

                    sibling_idx = if is_right_child {
                        self.storage[parent_idx].left
                    } else {
                        self.storage[parent_idx].right
                    };

                    if sibling_idx == Self::BLACK_NIL {
                        step = RosewoodDeletionStep::Ended;
                        continue;
                    }

                    assert_ne!(sibling_idx, Self::BLACK_NIL);

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
                    if sibling_idx != Self::BLACK_NIL {
                        self.storage[sibling_idx].color = NodeColor::Red;
                    }
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

                    distant_nephew_idx = sibling_idx;
                    sibling_idx = close_nephew_idx;

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

        self.storage[self.root].color = NodeColor::Black;
    }

    fn rotate_left(&mut self, center: usize) {
        debug_assert_ne!(
            center,
            Self::BLACK_NIL,
            "Attempted to left rotate around NIL node"
        );

        let grandparent_idx = self.storage[center].parent;
        let sibling_idx = self.storage[center].right;

        debug_assert_ne!(
            sibling_idx,
            Self::BLACK_NIL,
            "Attempted to left rotate around {center} with NIL sibling"
        );

        let c_idx = self.storage[sibling_idx].left;

        self.storage[center].right = c_idx;
        if c_idx != Self::BLACK_NIL {
            self.storage[c_idx].parent = center;
        }

        self.storage[sibling_idx].left = center;
        self.storage[center].parent = sibling_idx;
        self.storage[sibling_idx].parent = grandparent_idx;

        if grandparent_idx == Self::BLACK_NIL {
            self.root = sibling_idx;
        } else if self.storage[grandparent_idx].right == center {
            self.storage[grandparent_idx].right = sibling_idx;
        } else {
            self.storage[grandparent_idx].left = sibling_idx;
        }
    }

    fn rotate_right(&mut self, center: usize) {
        debug_assert_ne!(
            center,
            Self::BLACK_NIL,
            "Attempted to right rotate around NIL node"
        );

        let grandparent_idx = self.storage[center].parent;
        let sibling_idx = self.storage[center].left;

        debug_assert_ne!(
            sibling_idx,
            Self::BLACK_NIL,
            "Attempted to right rotate around {center} with NIL sibling"
        );

        let c_idx = self.storage[sibling_idx].right;

        self.storage[center].left = c_idx;
        if c_idx != Self::BLACK_NIL {
            self.storage[c_idx].parent = center;
        }

        self.storage[sibling_idx].right = center;
        self.storage[center].parent = sibling_idx;
        self.storage[sibling_idx].parent = grandparent_idx;

        if grandparent_idx == Self::BLACK_NIL {
            self.root = sibling_idx;
        } else if self.storage[grandparent_idx].right == center {
            self.storage[grandparent_idx].right = sibling_idx;
        } else {
            self.storage[grandparent_idx].left = sibling_idx;
        }
    }
}

impl<'a, K: Ord> IntoIterator for &'a Rosewood<K> {
    type Item = &'a K;

    type IntoIter = RosewoodSortedIterator<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Ord> IntoIterator for &'a mut Rosewood<K> {
    type Item = &'a mut K;

    type IntoIter = RosewoodSortedIteratorMut<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K: Default + PartialEq + Ord> Rosewood<K> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            storage: alloc::vec![RosewoodNode::default()],
            length: 0,
            free_nodes_head: 0,
            root: Self::BLACK_NIL,
        }
    }

    pub fn reserve(&mut self, cap: usize) {
        self.storage.reserve(cap);
    }

    pub fn extract_lower_bound(&mut self, target: &K) -> Option<K> {
        let lower_bound = self.lower_bound(target)?;
        let key = take(&mut self.storage[lower_bound].key);
        self.delete(lower_bound);

        Some(key)
    }
}

impl<K: Default + Ord> Default for Rosewood<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::Rosewood;
    use rand::prelude::*;

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

    #[test]
    pub fn random_order_insertion() {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..10000).collect();

        indices.shuffle(&mut rng);

        let mut tree = Rosewood::<usize>::new();

        for &idx in &indices {
            tree.insert(idx);
        }

        for idx in &indices {
            assert!(tree.contains(idx));
        }
    }

    #[test]
    pub fn random_order_insertion_then_deletions() {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..10).collect();

        indices.shuffle(&mut rng);

        let mut tree = Rosewood::<usize>::new();

        for &idx in &indices {
            tree.insert(idx);
        }

        let mut rng = rand::thread_rng();
        let range = rand::distributions::Uniform::new(0, 10);

        let deletion_indices: Vec<usize> = (0..5).map(|_| rng.sample(&range)).collect();

        for idx in 0..deletion_indices.len() {
            tree.remove(&deletion_indices[idx]);

            assert!(
                !tree.contains(&deletion_indices[idx]),
                "FAILED DELETION {} {:?} root = {}",
                deletion_indices[idx],
                tree.storage,
                tree.root
            );
        }
    }

    #[test]
    pub fn random_order_simultaneous_insertion_deletions() {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..100000).collect();

        indices.shuffle(&mut rng);

        let mut tree = Rosewood::<usize>::new();

        for &idx in &indices {
            tree.insert(idx);
        }

        let mut rng = rand::thread_rng();
        let range = rand::distributions::Uniform::new(0, 100000);

        let deletion_indices: Vec<usize> = (0..200000).map(|_| rng.sample(&range)).collect();
        let insertion_indices: Vec<usize> = (0..200000).map(|_| rng.sample(&range)).collect();

        for idx in 0..insertion_indices.len() {
            tree.insert(insertion_indices[idx]);

            tree.remove(&deletion_indices[idx]);
            assert!(
                deletion_indices[idx] == insertion_indices[idx]
                    || tree.contains(&insertion_indices[idx]),
                "FAILED INSERTION at insertion = {}, deletion = {}",
                insertion_indices[idx],
                deletion_indices[idx],
            );
            assert!(
                !tree.contains(&deletion_indices[idx]),
                "FAILED DELETION at insertion = {}, deletion = {}",
                insertion_indices[idx],
                deletion_indices[idx]
            );
        }
    }
}
