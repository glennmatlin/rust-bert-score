//! Benchmarks for rust-bert-score components.

use rust_bert_score::core::compute_bertscore;
use tch::{Device, Tensor};
// use test::Bencher;

use criterion::{criterion_group, criterion_main, Criterion};

/// Benchmark scoring (cossim) for CPU device.
fn bench_score_embeddings_cpu(c : &mut Criterion) {
    let device = Device::Cpu;
    let cand_emb = Tensor::randn(&[100, 768], (tch::Kind::Float, device));
    let ref_emb = Tensor::randn(&[100, 768], (tch::Kind::Float, device));
    let cand_mask = Tensor::ones(&[100], (tch::Kind::Float, device));
    let ref_mask = Tensor::ones(&[100], (tch::Kind::Float, device));

    c.bench_function("compute_bertscore", |b| {
        b.iter(|| compute_bertscore(&cand_emb, &ref_emb, &cand_mask, &ref_mask, None))
    });
}	

/// Benchmark scoring (cossim) for CUDA device.
/// Note this will be slower than the CPU version due to data transfer overhead
/// This is expected but we gain when actually using the model to generate embeddings
fn bench_score_embeddings_cuda(c : &mut Criterion) {
    let device = Device::Cuda(0);
    let cand_emb = Tensor::randn(&[100, 768], (tch::Kind::Float, device));
    let ref_emb = Tensor::randn(&[100, 768], (tch::Kind::Float, device));
    let cand_mask = Tensor::ones(&[100], (tch::Kind::Float, device));
    let ref_mask = Tensor::ones(&[100], (tch::Kind::Float, device));

    c.bench_function("compute_bertscore_cuda", |b| {
        b.iter(|| compute_bertscore(&cand_emb, &ref_emb, &cand_mask, &ref_mask, None))
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_score_embeddings_cpu,
        bench_score_embeddings_cuda,
);
criterion_main!(benches);
