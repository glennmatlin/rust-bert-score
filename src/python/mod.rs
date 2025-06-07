//! Python bindings for rust-bert-score.

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashSet;

use crate::{
    baseline::{BaselineManager, BaselineScores},
    similarity::compute_bertscore,
    BERTScoreResult, BERTScorer as RustBERTScorer, BERTScorerBuilder as RustBERTScorerBuilder,
    BERTScorerConfig as RustBERTScorerConfig,
};
use rust_bert::pipelines::common::ModelType;
use tch::{Device, Tensor};

/// Python wrapper for BERTScoreResult
#[pyclass]
#[derive(Clone)]
pub struct PyBERTScoreResult {
    #[pyo3(get)]
    pub precision: f32,
    #[pyo3(get)]
    pub recall: f32,
    #[pyo3(get)]
    pub f1: f32,
}

impl From<BERTScoreResult> for PyBERTScoreResult {
    fn from(result: BERTScoreResult) -> Self {
        Self {
            precision: result.precision,
            recall: result.recall,
            f1: result.f1,
        }
    }
}

#[pymethods]
impl PyBERTScoreResult {
    fn __repr__(&self) -> String {
        format!(
            "BERTScoreResult(precision={:.3}, recall={:.3}, f1={:.3})",
            self.precision, self.recall, self.f1
        )
    }
}

/// Python wrapper for BERTScorer
#[pyclass]
pub struct BERTScorer {
    inner: RustBERTScorer,
}

#[pymethods]
impl BERTScorer {
    /// Create a new BERTScorer with the given configuration.
    #[new]
    #[pyo3(signature = (
        model_type="roberta",
        model_name="roberta-large",
        vocab_path,
        merges_path=None,
        language="en",
        num_layers=None,
        batch_size=64,
        use_idf=false,
        rescale_with_baseline=false,
        device=None,
    ))]
    fn new(
        model_type: &str,
        model_name: &str,
        vocab_path: &str,
        merges_path: Option<&str>,
        language: &str,
        num_layers: Option<i32>,
        batch_size: usize,
        use_idf: bool,
        rescale_with_baseline: bool,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let model_type = match model_type.to_lowercase().as_str() {
            "bert" => ModelType::Bert,
            "distilbert" => ModelType::DistilBert,
            "roberta" => ModelType::Roberta,
            "deberta" => ModelType::Deberta,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown model type: {}",
                    model_type
                )))
            }
        };

        let device = match device {
            Some("cuda") => Device::cuda_if_available(),
            Some("cpu") => Device::Cpu,
            Some(d) if d.starts_with("cuda:") => {
                let device_id = d[5..]
                    .parse::<usize>()
                    .map_err(|_| PyValueError::new_err("Invalid CUDA device ID"))?;
                Device::Cuda(device_id)
            }
            None => Device::cuda_if_available(),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown device: {:?}",
                    device
                )))
            }
        };

        let config = RustBERTScorerConfig {
            model_type,
            model_name: model_name.to_string(),
            language: language.to_string(),
            vocab_path: vocab_path.to_string(),
            merges_path: merges_path.map(|s| s.to_string()),
            lower_case: model_name.contains("uncased"),
            device,
            num_layers,
            max_length: 512,
            batch_size,
            use_idf,
            rescale_with_baseline,
            custom_baseline: None,
        };

        let inner = RustBERTScorer::new(config)
            .map_err(|e| PyValueError::new_err(format!("Failed to create BERTScorer: {}", e)))?;

        Ok(Self { inner })
    }

    /// Score candidate sentences against reference sentences.
    fn score(
        &self,
        candidates: Vec<String>,
        references: Vec<String>,
    ) -> PyResult<Vec<PyBERTScoreResult>> {
        let results = self
            .inner
            .score(&candidates, &references)
            .map_err(|e| PyValueError::new_err(format!("Scoring failed: {}", e)))?;

        Ok(results.into_iter().map(PyBERTScoreResult::from).collect())
    }

    /// Score candidates against multiple references per candidate.
    fn score_multi_refs(
        &self,
        candidates: Vec<String>,
        references: Vec<Vec<String>>,
    ) -> PyResult<Vec<PyBERTScoreResult>> {
        let results = self
            .inner
            .score_multi_refs(&candidates, &references)
            .map_err(|e| PyValueError::new_err(format!("Multi-ref scoring failed: {}", e)))?;

        Ok(results.into_iter().map(PyBERTScoreResult::from).collect())
    }
}

/// Compute BERTScore from pre-computed embeddings.
#[pyfunction]
#[pyo3(signature = (
    candidate_embeddings,
    reference_embeddings,
    candidate_mask=None,
    reference_mask=None,
    candidate_idf_weights=None,
    reference_idf_weights=None,
))]
fn compute_bertscore_from_embeddings<'py>(
    py: Python<'py>,
    candidate_embeddings: &PyArray2<f32>,
    reference_embeddings: &PyArray2<f32>,
    candidate_mask: Option<&PyArray1<f32>>,
    reference_mask: Option<&PyArray1<f32>>,
    candidate_idf_weights: Option<&PyArray1<f32>>,
    reference_idf_weights: Option<&PyArray1<f32>>,
) -> PyResult<PyBERTScoreResult> {
    let cand_shape = candidate_embeddings.shape();
    let ref_shape = reference_embeddings.shape();

    // Convert numpy arrays to tensors
    let cand_emb = unsafe {
        Tensor::from_blob(
            candidate_embeddings.as_ptr(),
            &[cand_shape[0] as i64, cand_shape[1] as i64],
            &[cand_shape[1] as i64, 1],
            tch::Kind::Float,
            Device::Cpu,
        )
    };

    let ref_emb = unsafe {
        Tensor::from_blob(
            reference_embeddings.as_ptr(),
            &[ref_shape[0] as i64, ref_shape[1] as i64],
            &[ref_shape[1] as i64, 1],
            tch::Kind::Float,
            Device::Cpu,
        )
    };

    // Handle masks
    let cand_mask = match candidate_mask {
        Some(mask) => unsafe {
            Tensor::from_blob(
                mask.as_ptr(),
                &[cand_shape[0] as i64],
                &[1],
                tch::Kind::Float,
                Device::Cpu,
            )
        },
        None => Tensor::ones(&[cand_shape[0] as i64], (tch::Kind::Float, Device::Cpu)),
    };

    let ref_mask = match reference_mask {
        Some(mask) => unsafe {
            Tensor::from_blob(
                mask.as_ptr(),
                &[ref_shape[0] as i64],
                &[1],
                tch::Kind::Float,
                Device::Cpu,
            )
        },
        None => Tensor::ones(&[ref_shape[0] as i64], (tch::Kind::Float, Device::Cpu)),
    };

    // Handle IDF weights
    let idf_weights = match (candidate_idf_weights, reference_idf_weights) {
        (Some(cand_idf), Some(ref_idf)) => {
            let cand_tensor = unsafe {
                Tensor::from_blob(
                    cand_idf.as_ptr(),
                    &[cand_shape[0] as i64],
                    &[1],
                    tch::Kind::Float,
                    Device::Cpu,
                )
            };
            let ref_tensor = unsafe {
                Tensor::from_blob(
                    ref_idf.as_ptr(),
                    &[ref_shape[0] as i64],
                    &[1],
                    tch::Kind::Float,
                    Device::Cpu,
                )
            };
            Some((&cand_tensor, &ref_tensor))
        }
        _ => None,
    };

    let result = compute_bertscore(&cand_emb, &ref_emb, &cand_mask, &ref_mask, idf_weights)
        .map_err(|e| PyValueError::new_err(format!("BERTScore computation failed: {}", e)))?;

    Ok(PyBERTScoreResult::from(result))
}

/// Python wrapper for BaselineManager
#[pyclass]
pub struct PyBaselineManager {
    inner: BaselineManager,
}

#[pymethods]
impl PyBaselineManager {
    #[new]
    fn new() -> Self {
        Self {
            inner: BaselineManager::new(),
        }
    }

    /// Create a baseline manager with default baselines.
    #[staticmethod]
    fn with_defaults() -> Self {
        Self {
            inner: BaselineManager::with_defaults(),
        }
    }

    /// Add a baseline for a model and language.
    fn add_baseline(&mut self, model: &str, language: &str, precision: f32, recall: f32, f1: f32) {
        self.inner
            .add_baseline(model, language, BaselineScores::new(precision, recall, f1));
    }

    /// Load baselines from a TSV file.
    fn load_from_file(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .load_from_file(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to load baselines: {}", e)))
    }

    /// Rescale scores using baseline.
    fn rescale_scores(
        &self,
        model: &str,
        language: &str,
        precision: f32,
        recall: f32,
        f1: f32,
    ) -> Option<(f32, f32, f32)> {
        self.inner
            .rescale_scores(model, language, precision, recall, f1)
    }
}

/// Python module definition
#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BERTScorer>()?;
    m.add_class::<PyBERTScoreResult>()?;
    m.add_class::<PyBaselineManager>()?;
    m.add_function(wrap_pyfunction!(compute_bertscore_from_embeddings, m)?)?;
    Ok(())
}
