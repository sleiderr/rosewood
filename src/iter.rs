use core::{marker::PhantomData, ptr};

use alloc::vec::Vec;

use crate::{NodeIndex, Rosewood, RosewoodNode};

pub struct RosewoodSortedIterator<'a, K: Ord> {
    pub(crate) tree: &'a Rosewood<K>,
    pub(crate) curr: NodeIndex,
    pub(crate) stack: Vec<NodeIndex>,
}

impl<'a, K: Ord> Iterator for RosewoodSortedIterator<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr != Rosewood::<K>::BLACK_NIL {
            self.stack.push(self.curr);
            self.curr = self.tree.get_node_by_idx(self.curr).left_child();
        }

        if let Some(node) = self.stack.pop() {
            self.curr = self.tree.get_node_by_idx(node).right_child();

            return Some(&self.tree.get_node_by_idx(node).key);
        }

        None
    }
}

pub struct RosewoodSortedIteratorMut<'a, K: Ord> {
    pub(crate) tree: *mut Rosewood<K>,
    pub(crate) curr: NodeIndex,
    pub(crate) stack: Vec<NodeIndex>,
    pub(crate) phantom: PhantomData<&'a Rosewood<K>>,
}

impl<K: Ord> RosewoodSortedIteratorMut<'_, K> {
    fn get_node_mut(&mut self, node_idx: NodeIndex) -> &mut RosewoodNode<K> {
        unsafe { &mut self.tree.as_mut().unwrap().storage[node_idx.0] }
    }
}

impl<'a, K: Ord> Iterator for RosewoodSortedIteratorMut<'a, K> {
    type Item = &'a mut K;

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr != Rosewood::<K>::BLACK_NIL {
            self.stack.push(self.curr);
            self.curr = self.get_node_mut(self.curr).left_child();
        }

        if let Some(node) = self.stack.pop() {
            self.curr = self.get_node_mut(node).right_child();
            let key = unsafe { &mut (*self.tree).get_node_by_idx_mut(node).key };

            return Some(unsafe { &mut *(ptr::from_mut(key)) });
        }

        None
    }
}
