use crate::Rosewood;

struct MapEntry<K: Ord, V> {
    key: K,
    value: Option<V>,
}

impl<K: Default + Ord, V> Default for MapEntry<K, V> {
    fn default() -> Self {
        Self {
            key: K::default(),
            value: Option::default(),
        }
    }
}

impl<K: Ord, V> PartialEq for MapEntry<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<K: Ord, V> Eq for MapEntry<K, V> {}

impl<K: Ord, V> PartialOrd for MapEntry<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.key.cmp(&other.key))
    }
}

impl<K: Ord, V> Ord for MapEntry<K, V> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

/// An associative array, storing key-value pairs.
///
/// Uses a Rosewood red-black tree with a specialized key tyoe.
pub struct RosewoodMap<K: Ord, V> {
    tree: Rosewood<MapEntry<K, V>>,
}

impl<K: Default + Ord, V> RosewoodMap<K, V> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tree: Rosewood::new(),
        }
    }

    pub fn contains_key(&self, key: K) -> bool {
        self.tree.contains(&MapEntry { key, value: None })
    }

    pub fn insert(&mut self, key: K, value: V) -> bool {
        self.tree.insert(MapEntry {
            key,
            value: Some(value),
        })
    }

    pub fn get(&self, key: K) -> Option<&V> {
        let dummy_entry = MapEntry { key, value: None };

        self.tree
            .find_lower_bound(&dummy_entry)
            .filter(|&e| e.key == dummy_entry.key)?
            .value
            .as_ref()
    }

    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        let dummy_entry = MapEntry { key, value: None };

        self.tree
            .find_lower_bound_mut(&dummy_entry)
            .filter(|e| e.key == dummy_entry.key)?
            .value
            .as_mut()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.len() == 0
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tree.len()
    }
}

impl<K: Default + Ord, V> Default for RosewoodMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::RosewoodMap;

    #[test]
    pub fn map_entry_multi_insertion() {
        let mut map = RosewoodMap::<usize, usize>::new();

        map.insert(3, 17);
        map.insert(2, 12);
        map.insert(1, 7);

        assert!(map.contains_key(2));
        assert!(map.contains_key(1));
        assert!(map.contains_key(3));

        map.insert(3, 19);
        assert_eq!(*map.get(3).unwrap(), 17);
    }

    #[test]
    pub fn map_update_entry() {
        let mut map = RosewoodMap::<usize, usize>::new();

        map.insert(3, 17);
        *map.get_mut(3).unwrap() = 5;

        assert_eq!(*map.get(3).unwrap(), 5);
    }
}
