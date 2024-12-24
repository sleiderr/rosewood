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
use std::ops::{Add, Sub};

use alloc::vec::Vec;
use iter::{RosewoodSortedIterator, RosewoodSortedIteratorMut};

/*
store color information in the parent key ? reduces number of usable positions, but that should be fine in most
scenarios (with 64 bit usize, we can use 2^63 - 1 different keys).
other options: bitmap, bool in every node,
*/

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeIndex(pub usize);

impl Add<usize> for NodeIndex {
    type Output = NodeIndex;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl Sub<usize> for NodeIndex {
    type Output = NodeIndex;

    fn sub(self, rhs: usize) -> Self::Output {
        Self(self.0 - rhs)
    }
}

impl From<NodeIndex> for usize {
    #[inline]
    fn from(value: NodeIndex) -> Self {
        value.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Direction {
    Left = 0,
    Right = 1,
}

impl Direction {
    fn invert(self) -> Self {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

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
    parent: NodeIndex,
    children: [NodeIndex; 2],
}

impl<K> RosewoodNode<K> {
    #[inline]
    #[must_use]
    fn left_child(&self) -> NodeIndex {
        self.children[0]
    }

    #[inline]
    fn set_left_child(&mut self, child: NodeIndex) {
        self.children[0] = child;
    }

    #[inline]
    #[must_use]
    fn right_child(&self) -> NodeIndex {
        self.children[1]
    }

    #[inline]
    fn set_right_child(&mut self, child: NodeIndex) {
        self.children[1] = child;
    }

    #[inline]
    #[must_use]
    fn get_child_by_direction(&self, direction: Direction) -> NodeIndex {
        self.children[direction as usize]
    }

    #[inline]
    fn set_child_by_direction(&mut self, child: NodeIndex, direction: Direction) {
        self.children[direction as usize] = child
    }
}

impl<K> RosewoodNode<K> {
    fn new_isolated(key: K) -> Self {
        Self {
            key,
            color: NodeColor::default(),
            parent: NodeIndex::default(),
            children: [NodeIndex::default(); 2],
        }
    }
}

impl<K: Default> Default for RosewoodNode<K> {
    fn default() -> Self {
        Self {
            key: K::default(),
            color: NodeColor::default(),
            parent: NodeIndex::default(),
            children: [NodeIndex::default(); 2],
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
enum RosewoodInsertionStep {
    Running,
    UncleRed,
    ParentRedRoot,
    ParentRedUncleBlack,
    Ended,
}

#[derive(Debug)]
pub struct Rosewood<K: PartialEq + Ord> {
    storage: Vec<RosewoodNode<K>>,
    free_nodes_head: NodeIndex,
    length: usize,
    root: NodeIndex,
}

impl<K: PartialEq + Ord> Rosewood<K> {
    const BLACK_NIL: NodeIndex = NodeIndex(0);

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
        self.lower_bound(target)
            .map(|idx| &self.get_node_by_idx(idx).key)
    }

    pub fn find_lower_bound_mut(&mut self, target: &K) -> Option<&mut K> {
        self.lower_bound(target)
            .map(|idx| &mut self.get_node_by_idx_mut(idx).key)
    }

    pub fn insert(&mut self, key: K) -> bool {
        let mut current_node = self.root;
        let mut parent_node = Self::BLACK_NIL;

        while current_node != Self::BLACK_NIL {
            parent_node = current_node;
            let curr_node_storage = self.get_node_by_idx(current_node);

            match key.cmp(&curr_node_storage.key) {
                Ordering::Less => {
                    current_node = curr_node_storage.left_child();
                }
                Ordering::Equal => {
                    return false;
                }
                Ordering::Greater => {
                    current_node = curr_node_storage.right_child();
                }
            }
        }

        let new_node_pos = self.find_free_slot_and_fill(key);

        self.get_node_by_idx_mut(new_node_pos).parent = parent_node;
        self.get_node_by_idx_mut(new_node_pos)
            .set_left_child(Self::BLACK_NIL);
        self.get_node_by_idx_mut(new_node_pos)
            .set_right_child(Self::BLACK_NIL);
        self.get_node_by_idx_mut(new_node_pos).color = NodeColor::Red;
        self.length += 1;

        if parent_node == Self::BLACK_NIL {
            self.root = new_node_pos;
            return true;
        } else if self.get_node_by_idx(new_node_pos).key < self.get_node_by_idx(parent_node).key {
            self.get_node_by_idx_mut(parent_node)
                .set_left_child(new_node_pos);
        } else {
            self.get_node_by_idx_mut(parent_node)
                .set_right_child(new_node_pos);
        }

        self.fix_red_violation(new_node_pos);

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
            if self.get_node_by_idx(curr_elem).right_child() == Self::BLACK_NIL {
                return Some(&self.get_node_by_idx(curr_elem).key);
            }

            curr_elem = self.get_node_by_idx(curr_elem).right_child();
        }
    }

    #[must_use]
    pub fn min(&self) -> Option<&K> {
        let mut curr_elem = self.root;

        if curr_elem == Self::BLACK_NIL {
            return None;
        }

        loop {
            if self.get_node_by_idx(curr_elem).left_child() == Self::BLACK_NIL {
                return Some(&self.get_node_by_idx(curr_elem).key);
            }

            curr_elem = self.get_node_by_idx(curr_elem).left_child();
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
    fn get_node_color(&self, idx: NodeIndex) -> NodeColor {
        self.storage[idx.0].color
    }

    #[inline]
    fn set_node_color(&mut self, idx: NodeIndex, color: NodeColor) {
        self.storage[idx.0].color = color;
    }

    #[inline]
    fn get_node_by_idx(&self, idx: NodeIndex) -> &RosewoodNode<K> {
        &self.storage[idx.0]
    }

    #[inline]
    fn get_node_by_idx_mut(&mut self, idx: NodeIndex) -> &mut RosewoodNode<K> {
        &mut self.storage[idx.0]
    }

    #[inline]
    #[must_use]
    fn available_slots(&self) -> usize {
        self.storage.len() - self.length - 1
    }

    fn shrink(&mut self) {
        let mid = (self.storage.len() >> 1) + 1;
        let (front, back) = self.storage.split_at_mut(usize::from(mid));

        let mut front_idx = 1;

        for idx in 0..back.len() {
            if matches!(back[idx].color, NodeColor::Black | NodeColor::Red) {
                loop {
                    if matches!(front[front_idx].color, NodeColor::Free) {
                        swap(&mut front[front_idx], &mut back[idx]);
                        let parent_idx = front[front_idx].parent;
                        let left_idx = front[front_idx].left_child();
                        let right_idx = front[front_idx].right_child();

                        if self.root == NodeIndex(mid + idx) {
                            self.root = NodeIndex(front_idx);
                        }

                        if parent_idx != Self::BLACK_NIL {
                            if parent_idx >= NodeIndex(mid) {
                                if back[usize::from(parent_idx - mid)].right_child()
                                    == NodeIndex(mid + idx)
                                {
                                    back[usize::from(parent_idx - mid)]
                                        .set_right_child(NodeIndex(front_idx))
                                } else {
                                    back[usize::from(parent_idx - mid)]
                                        .set_left_child(NodeIndex(front_idx))
                                }
                            } else {
                                if front[usize::from(parent_idx)].right_child()
                                    == NodeIndex(mid + idx)
                                {
                                    front[usize::from(parent_idx)]
                                        .set_right_child(NodeIndex(front_idx));
                                } else {
                                    front[usize::from(parent_idx)]
                                        .set_left_child(NodeIndex(front_idx));
                                }
                            }
                        }

                        if left_idx != Self::BLACK_NIL {
                            if left_idx >= NodeIndex(mid) {
                                back[usize::from(left_idx - mid)].parent = NodeIndex(front_idx);
                            } else {
                                front[usize::from(left_idx)].parent = NodeIndex(front_idx);
                            }
                        }

                        if right_idx != Self::BLACK_NIL {
                            if right_idx >= NodeIndex(mid) {
                                back[usize::from(right_idx - mid)].parent = NodeIndex(front_idx);
                            } else {
                                front[usize::from(right_idx)].parent = NodeIndex(front_idx);
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
        self.free_nodes_head = Self::BLACK_NIL;

        for (idx, slot) in self.storage.iter_mut().enumerate() {
            if matches!(slot.color, NodeColor::Free) {
                swap(&mut self.free_nodes_head, &mut slot.parent);
                self.free_nodes_head = NodeIndex(idx);
            }
        }
    }

    fn lower_bound(&self, target: &K) -> Option<NodeIndex> {
        let mut walker = self.root;
        let mut best_fit_idx: Option<NodeIndex> = None;

        while walker != Self::BLACK_NIL {
            let walker_key = &self.get_node_by_idx(walker).key;

            match walker_key.cmp(target) {
                Ordering::Less => {
                    walker = self.get_node_by_idx(walker).right_child();
                }
                Ordering::Equal => {
                    return Some(walker);
                }
                Ordering::Greater => {
                    best_fit_idx = match best_fit_idx {
                        Some(idx) => {
                            if let Ordering::Less = walker_key.cmp(&self.get_node_by_idx(idx).key) {
                                Some(walker)
                            } else {
                                Some(idx)
                            }
                        }
                        None => Some(walker),
                    };
                    walker = self.get_node_by_idx(walker).left_child();
                }
            }
        }

        best_fit_idx
    }

    #[inline]
    fn insert_free_slot(&mut self, slot: NodeIndex) {
        self.get_node_by_idx_mut(slot).parent = self.free_nodes_head;
        self.get_node_by_idx_mut(slot).color = NodeColor::Free;

        self.free_nodes_head = slot;
    }

    #[inline]
    fn find_free_slot_and_fill(&mut self, key: K) -> NodeIndex {
        if self.free_nodes_head == Self::BLACK_NIL {
            self.storage.push(RosewoodNode::new_isolated(key));

            NodeIndex(self.storage.len() - 1)
        } else {
            let free_slot = self.free_nodes_head;

            self.get_node_by_idx_mut(free_slot).key = key;
            self.free_nodes_head = self.get_node_by_idx(free_slot).parent;

            free_slot
        }
    }

    fn lookup(&self, key: &K) -> NodeIndex {
        let mut current_node = self.root;

        while current_node != Self::BLACK_NIL {
            let curr_node_storage = self.get_node_by_idx(current_node);

            match key.cmp(&curr_node_storage.key) {
                Ordering::Less => {
                    current_node = curr_node_storage.left_child();
                }
                Ordering::Equal => {
                    return current_node;
                }
                Ordering::Greater => {
                    current_node = curr_node_storage.right_child();
                }
            }
        }

        Self::BLACK_NIL
    }

    fn delete(&mut self, node_idx: NodeIndex) {
        self.length -= 1;

        match (
            self.get_node_by_idx(node_idx).left_child(),
            self.get_node_by_idx(node_idx).right_child(),
        ) {
            (Self::BLACK_NIL, Self::BLACK_NIL) => {
                if self.root == node_idx {
                    self.root = Self::BLACK_NIL;
                    self.insert_free_slot(node_idx);
                } else if matches!(self.get_node_color(node_idx), NodeColor::Red) {
                    let parent_idx = self.get_node_by_idx(node_idx).parent;
                    let parent = self.get_node_by_idx_mut(parent_idx);

                    if parent.left_child() == node_idx {
                        parent.set_left_child(Self::BLACK_NIL);
                    } else {
                        parent.set_right_child(Self::BLACK_NIL);
                    }

                    self.insert_free_slot(node_idx);
                } else {
                    self.delete_black_leaf(node_idx);
                    self.insert_free_slot(node_idx);
                }
            }
            (single_child_idx, Self::BLACK_NIL) | (Self::BLACK_NIL, single_child_idx) => {
                let parent_idx = self.get_node_by_idx(node_idx).parent;
                let parent = self.get_node_by_idx_mut(parent_idx);

                if parent_idx != Self::BLACK_NIL {
                    if parent.left_child() == node_idx {
                        parent.set_left_child(single_child_idx);
                    } else {
                        parent.set_right_child(single_child_idx);
                    }
                }

                if self.root == node_idx {
                    self.root = single_child_idx;
                }

                self.get_node_by_idx_mut(single_child_idx).parent = parent_idx;
                self.set_node_color(single_child_idx, NodeColor::Black);

                self.insert_free_slot(node_idx);
            }
            (left_child_idx, _right_child_idx) => {
                self.length += 1;
                let succ = self.find_inorder_predecessor(left_child_idx);
                self.swap_nodes(succ, node_idx);

                self.delete(succ);
            }
        }

        debug_assert_eq!(self.get_node_color(Self::BLACK_NIL), NodeColor::Black);
    }

    fn swap_nodes(&mut self, node_a: NodeIndex, node_b: NodeIndex) {
        if node_a == node_b {
            return;
        }

        let (part_a, part_b) = self.storage.split_at_mut(min(node_a.0, node_b.0) + 1);

        swap(
            &mut part_a[part_a.len() - 1].key,
            &mut part_b[max(node_a.0, node_b.0) - min(node_a.0, node_b.0) - 1].key,
        );
    }

    fn find_inorder_predecessor(&self, subtree_root: NodeIndex) -> NodeIndex {
        let mut curr_node = subtree_root;

        loop {
            let next_node = self.get_node_by_idx(curr_node).right_child();

            if next_node == Self::BLACK_NIL {
                return curr_node;
            }
            curr_node = next_node;
        }
    }

    fn delete_black_leaf(&mut self, node_idx: NodeIndex) {
        let mut step = RosewoodDeletionStep::Starting;
        let mut curr_node = node_idx;
        let mut parent_idx = self.get_node_by_idx(curr_node).parent;
        let mut child_direction = if self.get_node_by_idx(parent_idx).right_child() == curr_node {
            Direction::Right
        } else {
            Direction::Left
        };
        let (mut sibling_idx, mut distant_nephew_idx, mut close_nephew_idx) =
            (Self::BLACK_NIL, Self::BLACK_NIL, Self::BLACK_NIL);

        loop {
            match step {
                RosewoodDeletionStep::Starting => {
                    self.get_node_by_idx_mut(parent_idx)
                        .set_child_by_direction(Self::BLACK_NIL, child_direction);

                    step = RosewoodDeletionStep::UpdateVariables;
                }
                RosewoodDeletionStep::Continue => {
                    child_direction = if self.get_node_by_idx(parent_idx).right_child() == curr_node
                    {
                        Direction::Right
                    } else {
                        Direction::Left
                    };
                    step = RosewoodDeletionStep::UpdateVariables;
                }
                RosewoodDeletionStep::UpdateVariables => {
                    parent_idx = self.get_node_by_idx(curr_node).parent;

                    (sibling_idx, distant_nephew_idx, close_nephew_idx) = {
                        let sibling_idx = self
                            .get_node_by_idx(parent_idx)
                            .get_child_by_direction(child_direction.invert());

                        (
                            sibling_idx,
                            self.get_node_by_idx(sibling_idx)
                                .get_child_by_direction(child_direction.invert()),
                            self.get_node_by_idx(sibling_idx)
                                .get_child_by_direction(child_direction),
                        )
                    };

                    if parent_idx == Self::BLACK_NIL {
                        step = RosewoodDeletionStep::Ended;
                        continue;
                    }

                    if matches!(self.get_node_color(sibling_idx), NodeColor::Red) {
                        step = RosewoodDeletionStep::RedSibling;
                        continue;
                    }

                    if distant_nephew_idx != Self::BLACK_NIL
                        && matches!(self.get_node_color(distant_nephew_idx), NodeColor::Red)
                    {
                        step = RosewoodDeletionStep::DistantRed;
                        continue;
                    }

                    if close_nephew_idx != Self::BLACK_NIL
                        && matches!(self.get_node_color(close_nephew_idx), NodeColor::Red)
                    {
                        step = RosewoodDeletionStep::CloseRedDistantBlack;
                        continue;
                    }

                    if matches!(self.get_node_color(parent_idx), NodeColor::Red) {
                        step = RosewoodDeletionStep::ParentRedChildrenBlack;
                        continue;
                    }

                    step = RosewoodDeletionStep::ParentBlackSiblingBlackChildrenBlack;
                    continue;
                }
                RosewoodDeletionStep::ParentBlackSiblingBlackChildrenBlack => {
                    if sibling_idx != Self::BLACK_NIL {
                        self.set_node_color(sibling_idx, NodeColor::Red);
                    }
                    curr_node = parent_idx;

                    step = RosewoodDeletionStep::Continue;
                }
                RosewoodDeletionStep::RedSibling => {
                    self.rotate(parent_idx, child_direction);

                    self.set_node_color(parent_idx, NodeColor::Red);
                    self.set_node_color(sibling_idx, NodeColor::Black);

                    sibling_idx = self
                        .get_node_by_idx(parent_idx)
                        .get_child_by_direction(child_direction.invert());

                    if sibling_idx == Self::BLACK_NIL {
                        step = RosewoodDeletionStep::Ended;
                        continue;
                    }

                    assert_ne!(sibling_idx, Self::BLACK_NIL);

                    distant_nephew_idx = self
                        .get_node_by_idx(sibling_idx)
                        .get_child_by_direction(child_direction.invert());

                    step = RosewoodDeletionStep::ParentRedChildrenBlack;

                    if distant_nephew_idx != Self::BLACK_NIL
                        && matches!(self.get_node_color(distant_nephew_idx), NodeColor::Red)
                    {
                        step = RosewoodDeletionStep::DistantRed;
                    }

                    close_nephew_idx = self
                        .get_node_by_idx(sibling_idx)
                        .get_child_by_direction(child_direction);

                    if close_nephew_idx != Self::BLACK_NIL
                        && matches!(self.get_node_color(close_nephew_idx), NodeColor::Red)
                    {
                        step = RosewoodDeletionStep::CloseRedDistantBlack;
                    }

                    continue;
                }
                RosewoodDeletionStep::ParentRedChildrenBlack => {
                    if sibling_idx != Self::BLACK_NIL {
                        self.set_node_color(sibling_idx, NodeColor::Red);
                    }
                    self.set_node_color(parent_idx, NodeColor::Black);

                    step = RosewoodDeletionStep::Ended;
                    continue;
                }
                RosewoodDeletionStep::CloseRedDistantBlack => {
                    self.rotate(sibling_idx, child_direction.invert());

                    self.set_node_color(sibling_idx, NodeColor::Red);
                    self.set_node_color(close_nephew_idx, NodeColor::Black);

                    distant_nephew_idx = sibling_idx;
                    sibling_idx = close_nephew_idx;

                    step = RosewoodDeletionStep::DistantRed;
                    continue;
                }
                RosewoodDeletionStep::DistantRed => {
                    self.rotate(parent_idx, child_direction);

                    self.set_node_color(sibling_idx, self.get_node_color(parent_idx));
                    self.set_node_color(parent_idx, NodeColor::Black);
                    self.set_node_color(distant_nephew_idx, NodeColor::Black);

                    step = RosewoodDeletionStep::Ended;
                    continue;
                }
                RosewoodDeletionStep::Ended => {
                    return;
                }
            }
        }
    }

    fn fix_red_violation(&mut self, start_node_idx: NodeIndex) {
        let mut step = RosewoodInsertionStep::Running;
        let mut curr_node = start_node_idx;
        let mut uncle_idx = Self::BLACK_NIL;
        let mut parent_idx = Self::BLACK_NIL;
        let mut parent_child_direction = Direction::Left;
        let mut grandparent_idx = Self::BLACK_NIL;

        loop {
            match step {
                RosewoodInsertionStep::Running => {
                    parent_idx = self.get_node_by_idx(curr_node).parent;
                    if matches!(self.get_node_color(parent_idx), NodeColor::Black) {
                        step = RosewoodInsertionStep::Ended;
                        continue;
                    }

                    grandparent_idx = self.get_node_by_idx(parent_idx).parent;

                    if grandparent_idx == Self::BLACK_NIL {
                        step = RosewoodInsertionStep::ParentRedRoot;
                        continue;
                    }

                    let grandparent = self.get_node_by_idx(grandparent_idx);
                    parent_child_direction = if grandparent.right_child() == parent_idx {
                        Direction::Right
                    } else {
                        Direction::Left
                    };

                    uncle_idx = grandparent.get_child_by_direction(parent_child_direction.invert());

                    if matches!(self.get_node_color(uncle_idx), NodeColor::Black) {
                        step = RosewoodInsertionStep::ParentRedUncleBlack;
                    } else {
                        step = RosewoodInsertionStep::UncleRed;
                    }
                }
                RosewoodInsertionStep::UncleRed => {
                    self.set_node_color(parent_idx, NodeColor::Black);
                    self.set_node_color(uncle_idx, NodeColor::Black);
                    self.set_node_color(grandparent_idx, NodeColor::Red);

                    curr_node = grandparent_idx;
                    step = RosewoodInsertionStep::Running;
                }
                RosewoodInsertionStep::ParentRedRoot => {
                    self.set_node_color(parent_idx, NodeColor::Black);
                    step = RosewoodInsertionStep::Ended;
                }
                RosewoodInsertionStep::ParentRedUncleBlack => {
                    let parent = self.get_node_by_idx(parent_idx);
                    if parent.get_child_by_direction(parent_child_direction.invert()) == curr_node {
                        self.rotate(parent_idx, parent_child_direction);

                        curr_node = parent_idx;
                        parent_idx = self
                            .get_node_by_idx(grandparent_idx)
                            .get_child_by_direction(parent_child_direction);
                    }

                    self.set_node_color(parent_idx, NodeColor::Black);
                    self.set_node_color(grandparent_idx, NodeColor::Red);

                    self.rotate(grandparent_idx, parent_child_direction.invert());

                    step = RosewoodInsertionStep::Ended;
                }
                RosewoodInsertionStep::Ended => {
                    return;
                }
            }
        }
    }

    fn rotate(&mut self, center: NodeIndex, direction: Direction) {
        debug_assert_ne!(
            center,
            Self::BLACK_NIL,
            "Attempted to left rotate around NIL node"
        );

        let other_direction = direction.invert();

        let grandparent_idx = self.get_node_by_idx(center).parent;
        let sibling_idx = self
            .get_node_by_idx(center)
            .get_child_by_direction(other_direction);

        debug_assert_ne!(
            sibling_idx,
            Self::BLACK_NIL,
            "Attempted to left rotate around {center:?} with NIL sibling"
        );

        let c_idx = self
            .get_node_by_idx(sibling_idx)
            .get_child_by_direction(direction);

        self.get_node_by_idx_mut(center)
            .set_child_by_direction(c_idx, other_direction);

        if c_idx != Self::BLACK_NIL {
            self.get_node_by_idx_mut(c_idx).parent = center;
        }

        self.get_node_by_idx_mut(sibling_idx)
            .set_child_by_direction(center, direction);

        self.get_node_by_idx_mut(center).parent = sibling_idx;
        self.get_node_by_idx_mut(sibling_idx).parent = grandparent_idx;

        if grandparent_idx == Self::BLACK_NIL {
            self.root = sibling_idx;
        } else if self.get_node_by_idx(grandparent_idx).right_child() == center {
            self.get_node_by_idx_mut(grandparent_idx)
                .set_right_child(sibling_idx);
        } else {
            self.get_node_by_idx_mut(grandparent_idx)
                .set_left_child(sibling_idx);
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
            free_nodes_head: Self::BLACK_NIL,
            root: Self::BLACK_NIL,
        }
    }

    pub fn reserve(&mut self, cap: usize) {
        self.storage.reserve(cap);
    }

    pub fn extract_lower_bound(&mut self, target: &K) -> Option<K> {
        let lower_bound = self.lower_bound(target)?;
        let key = take(&mut self.get_node_by_idx_mut(lower_bound).key);
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

        assert_eq!(tree.get_node_by_idx(tree.root).key, 4);
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
                "FAILED DELETION {:?} {:?} root = {:?}",
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
