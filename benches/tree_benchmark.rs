use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use rosewood::Rosewood;
use std::collections::BTreeSet;

fn bench_baseline_multi_insertions(data: Vec<usize>) {
    let mut tree = BTreeSet::new();

    for i in data {
        tree.insert(i);
    }
}

fn bench_multi_insertions(data: Vec<usize>) {
    let mut tree = Rosewood::new();

    for i in data {
        tree.insert(i);
    }
}

fn bench_multi_insertions_hint(data: Vec<usize>) {
    let mut tree = Rosewood::new();
    tree.reserve(data.len());

    for i in data {
        tree.insert(i);
    }
}

fn init_large_btree() -> BTreeSet<usize> {
    let mut tree = BTreeSet::new();

    for i in random_insertion_order() {
        tree.insert(i);
    }

    tree
}

fn init_large_rosewood_tree() -> Rosewood<usize> {
    let mut tree = Rosewood::new();

    for i in random_insertion_order() {
        tree.insert(i);
    }

    tree
}

fn random_insertion_order() -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..100000).collect();

    indices.shuffle(&mut rng);

    indices
}

fn init_random_data() -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let range = rand::distributions::Uniform::new(0, 100000);

    let indices: Vec<usize> = (0..5000).map(|_| rng.sample(&range)).collect();

    indices
}

fn bench_baseline_random_deletions(mut tree: BTreeSet<usize>, indices: Vec<usize>) {
    for idx in indices {
        tree.remove(&idx);
    }
}

fn bench_random_deletions(mut tree: Rosewood<usize>, indices: Vec<usize>) {
    for idx in indices {
        tree.remove(&idx);
    }
}

fn bench_baseline_random_lookups(mut tree: BTreeSet<usize>, indices: Vec<usize>) {
    for idx in indices {
        assert!(tree.contains(&idx));
    }
}

fn bench_random_lookups(mut tree: Rosewood<usize>, indices: Vec<usize>) {
    for idx in indices {
        assert!(tree.contains(&idx));
    }
}

fn rosewood_tree_benchmark(c: &mut Criterion) {
    c.bench_function("baseline tree 100K insertions", |b| {
        b.iter_batched(
            || random_insertion_order(),
            |order| bench_baseline_multi_insertions(order),
            BatchSize::LargeInput,
        )
    });

    c.bench_function("tree 100K insertions", |b| {
        b.iter_batched(
            || random_insertion_order(),
            |order| bench_multi_insertions(order),
            BatchSize::LargeInput,
        )
    });

    c.bench_function("tree 100K insertions with size hint", |b| {
        b.iter_batched(
            || random_insertion_order(),
            |order| bench_multi_insertions_hint(order),
            BatchSize::LargeInput,
        )
    });

    c.bench_function("baseline tree random lookups", |b| {
        b.iter_batched(
            || (init_large_btree(), init_random_data()),
            |(tree, indices)| bench_baseline_random_lookups(tree, indices),
            BatchSize::LargeInput,
        )
    });

    c.bench_function("tree random lookups", |b| {
        b.iter_batched(
            || (init_large_rosewood_tree(), init_random_data()),
            |(tree, indices)| bench_random_lookups(tree, indices),
            BatchSize::LargeInput,
        )
    });

    c.bench_function("baseline tree random deletions", |b| {
        b.iter_batched(
            || (init_large_btree(), init_random_data()),
            |(tree, indices)| bench_baseline_random_deletions(tree, indices),
            BatchSize::LargeInput,
        )
    });

    c.bench_function("tree random deletions", |b| {
        b.iter_batched(
            || (init_large_rosewood_tree(), init_random_data()),
            |(tree, indices)| bench_random_deletions(tree, indices),
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, rosewood_tree_benchmark);
criterion_main!(benches);
