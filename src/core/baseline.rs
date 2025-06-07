//! Baseline rescaling module for score normalization.

use crate::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Baseline scores for a specific model and language combination.
#[derive(Debug, Clone, Copy)]
pub struct BaselineScores {
    /// Baseline precision score
    pub precision: f32,
    /// Baseline recall score
    pub recall: f32,
    /// Baseline F1 score
    pub f1: f32,
}

impl BaselineScores {
    /// Creates new baseline scores.
    pub fn new(precision: f32, recall: f32, f1: f32) -> Self {
        Self {
            precision,
            recall,
            f1,
        }
    }

    /// Rescales a score using the baseline.
    /// Formula: score' = (score - baseline) / (1 - baseline)
    fn rescale_score(score: f32, baseline: f32) -> f32 {
        if baseline >= 1.0 {
            // If baseline is 1.0, can't rescale (would divide by zero)
            score
        } else {
            (score - baseline) / (1.0 - baseline)
        }
    }

    /// Rescales precision, recall, and F1 scores using these baselines.
    pub fn rescale(&self, precision: f32, recall: f32, f1: f32) -> (f32, f32, f32) {
        let rescaled_precision = Self::rescale_score(precision, self.precision);
        let rescaled_recall = Self::rescale_score(recall, self.recall);
        let rescaled_f1 = Self::rescale_score(f1, self.f1);

        (rescaled_precision, rescaled_recall, rescaled_f1)
    }
}

/// Manager for baseline scores across different models and languages.
#[derive(Debug, Clone)]
pub struct BaselineManager {
    /// Map from (model_name, language) to baseline scores
    baselines: HashMap<(String, String), BaselineScores>,
}

impl BaselineManager {
    /// Creates an empty baseline manager.
    pub fn new() -> Self {
        Self {
            baselines: HashMap::new(),
        }
    }

    /// Creates a baseline manager with some common precomputed baselines.
    pub fn with_defaults() -> Self {
        let mut manager = Self::new();

        // Add some common baselines (these are example values - real ones would come from files)
        // English baselines
        manager.add_baseline(
            "bert-base-uncased",
            "en",
            BaselineScores::new(0.85, 0.85, 0.85),
        );
        manager.add_baseline("roberta-large", "en", BaselineScores::new(0.87, 0.87, 0.87));
        manager.add_baseline(
            "microsoft/deberta-xlarge-mnli",
            "en",
            BaselineScores::new(0.88, 0.88, 0.88),
        );

        // Multilingual baselines
        manager.add_baseline(
            "bert-base-multilingual-cased",
            "zh",
            BaselineScores::new(0.84, 0.84, 0.84),
        );
        manager.add_baseline(
            "xlm-roberta-large",
            "en",
            BaselineScores::new(0.86, 0.86, 0.86),
        );
        manager.add_baseline(
            "xlm-roberta-large",
            "zh",
            BaselineScores::new(0.85, 0.85, 0.85),
        );

        manager
    }

    /// Adds a baseline for a specific model and language.
    pub fn add_baseline(&mut self, model: &str, language: &str, scores: BaselineScores) {
        self.baselines
            .insert((model.to_string(), language.to_string()), scores);
    }

    /// Gets baseline scores for a model and language.
    pub fn get_baseline(&self, model: &str, language: &str) -> Option<&BaselineScores> {
        self.baselines
            .get(&(model.to_string(), language.to_string()))
    }

    /// Loads baselines from a TSV file.
    /// Expected format: model<tab>language<tab>precision<tab>recall<tab>f1
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let content = fs::read_to_string(path)?;

        for line in content.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 5 {
                let model = parts[0];
                let language = parts[1];
                let precision: f32 = parts[2].parse()?;
                let recall: f32 = parts[3].parse()?;
                let f1: f32 = parts[4].parse()?;

                self.add_baseline(model, language, BaselineScores::new(precision, recall, f1));
            }
        }

        Ok(())
    }

    /// Rescales scores using the appropriate baseline.
    /// Returns None if no baseline is found for the model/language.
    pub fn rescale_scores(
        &self,
        model: &str,
        language: &str,
        precision: f32,
        recall: f32,
        f1: f32,
    ) -> Option<(f32, f32, f32)> {
        self.get_baseline(model, language)
            .map(|baseline| baseline.rescale(precision, recall, f1))
    }
}

impl Default for BaselineManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Rescales BERTScore results using baseline values.
///
/// # Arguments
/// * `scores` - Raw BERTScore results (precision, recall, F1)
/// * `baseline` - Baseline scores for the model/language
///
/// # Returns
/// Rescaled scores where baseline maps to 0 and perfect score maps to 1
pub fn rescale_with_baseline(
    scores: (f32, f32, f32),
    baseline: &BaselineScores,
) -> (f32, f32, f32) {
    baseline.rescale(scores.0, scores.1, scores.2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_rescaling() {
        let baseline = BaselineScores::new(0.8, 0.8, 0.8);

        // Test rescaling with baseline 0.8
        let (p, r, f1) = baseline.rescale(0.9, 0.9, 0.9);

        // (0.9 - 0.8) / (1.0 - 0.8) = 0.1 / 0.2 = 0.5
        assert!((p - 0.5).abs() < 1e-6);
        assert!((r - 0.5).abs() < 1e-6);
        assert!((f1 - 0.5).abs() < 1e-6);

        // Test perfect score
        let (p, r, f1) = baseline.rescale(1.0, 1.0, 1.0);
        assert!((p - 1.0).abs() < 1e-6);
        assert!((r - 1.0).abs() < 1e-6);
        assert!((f1 - 1.0).abs() < 1e-6);

        // Test baseline score
        let (p, r, f1) = baseline.rescale(0.8, 0.8, 0.8);
        assert!((p - 0.0).abs() < 1e-6);
        assert!((r - 0.0).abs() < 1e-6);
        assert!((f1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_cases() {
        // Test with baseline close to 1.0
        let baseline = BaselineScores::new(0.99, 0.99, 0.99);
        let (p, _r, _f1) = baseline.rescale(0.995, 0.995, 0.995);

        // (0.995 - 0.99) / (1.0 - 0.99) = 0.005 / 0.01 = 0.5
        assert!((p - 0.5).abs() < 1e-4);

        // Test with baseline = 1.0 (no rescaling possible)
        let baseline = BaselineScores::new(1.0, 1.0, 1.0);
        let (p, r, f1) = baseline.rescale(0.9, 0.9, 0.9);
        assert_eq!(p, 0.9);
        assert_eq!(r, 0.9);
        assert_eq!(f1, 0.9);
    }

    #[test]
    fn test_baseline_manager() {
        let mut manager = BaselineManager::new();
        manager.add_baseline("bert-base", "en", BaselineScores::new(0.85, 0.85, 0.85));

        // Test retrieval
        let baseline = manager.get_baseline("bert-base", "en").unwrap();
        assert_eq!(baseline.precision, 0.85);

        // Test rescaling through manager
        let rescaled = manager.rescale_scores("bert-base", "en", 0.95, 0.93, 0.94);
        assert!(rescaled.is_some());

        let (p, r, f1) = rescaled.unwrap();
        // (0.95 - 0.85) / (1.0 - 0.85) = 0.1 / 0.15 = 0.667
        assert!((p - 0.667).abs() < 0.01);
        // (0.93 - 0.85) / 0.15 = 0.533
        assert!((r - 0.533).abs() < 0.01);
        // (0.94 - 0.85) / 0.15 = 0.6
        assert!((f1 - 0.6).abs() < 0.01);

        // Test missing baseline
        let rescaled = manager.rescale_scores("unknown-model", "en", 0.9, 0.9, 0.9);
        assert!(rescaled.is_none());
    }

    #[test]
    fn test_with_defaults() {
        let manager = BaselineManager::with_defaults();

        // Check that some defaults exist
        assert!(manager.get_baseline("bert-base-uncased", "en").is_some());
        assert!(manager.get_baseline("roberta-large", "en").is_some());
        assert!(manager.get_baseline("xlm-roberta-large", "zh").is_some());
    }
}
