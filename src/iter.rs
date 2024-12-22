use core::marker::PhantomData;

use alloc::vec::Vec;

use crate::{Rosewood, RosewoodNode};

pub struct RosewoodSortedIterator<'a, K: Ord> {
    pub(crate) tree: &'a Rosewood<K>,
    pub(crate) curr: usize,
    pub(crate) stack: Vec<usize>,
}

impl<'a, K: Ord> Iterator for RosewoodSortedIterator<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr != Rosewood::<K>::BLACK_NIL {
            self.stack.push(self.curr);
            self.curr = self.tree.storage[self.curr].left;
        }

        if let Some(node) = self.stack.pop() {
            self.curr = self.tree.storage[node].right;

            return Some(&self.tree.storage[node].key);
        }

        None
    }
}

pub struct RosewoodSortedIteratorMut<'a, K: Ord> {
    pub(crate) tree: *mut Rosewood<K>,
    pub(crate) curr: usize,
    pub(crate) stack: Vec<usize>,
    pub(crate) phantom: PhantomData<&'a Rosewood<K>>,
}

impl<'a, K: Ord> RosewoodSortedIteratorMut<'a, K> {
    fn get_node_mut(&mut self, node_idx: usize) -> &mut RosewoodNode<K> {
        unsafe { &mut self.tree.as_mut().unwrap().storage[node_idx] }
    }
}

impl<'a, K: Ord> Iterator for RosewoodSortedIteratorMut<'a, K> {
    type Item = &'a mut K;

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr != Rosewood::<K>::BLACK_NIL {
            self.stack.push(self.curr);
            self.curr = self.get_node_mut(self.curr).left;
        }

        if let Some(node) = self.stack.pop() {
            self.curr = self.get_node_mut(node).right;
            let key = unsafe { &mut (*self.tree).storage[node].key };

            return Some(unsafe { &mut *(key as *mut _) });
        }

        None
    }
}
