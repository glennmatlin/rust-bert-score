//! Command-line interface for rust-bert-score.
use clap::{Args, Parser, Subcommand, ValueEnum};
use rust_bert::pipelines::common::ModelType;

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CliEncoderModel {
    Bert,
    Distilbert,
    Roberta,
    Deberta,
}

impl From<CliEncoderModel> for ModelType {
    fn from(model: CliEncoderModel) -> Self {
        match model {
            CliEncoderModel::Bert => ModelType::Bert,
            CliEncoderModel::Distilbert => ModelType::DistilBert,
            CliEncoderModel::Roberta => ModelType::Roberta,
            CliEncoderModel::Deberta => ModelType::Deberta,
        }
    }
}

#[derive(Args)]
pub struct ScoreArgs {
    /// File containing candidate sentences (mutually exclusive with --input-tsv)
    #[arg(short, long, conflicts_with = "input_tsv")]
    pub candidates: Option<String>,

    /// File containing reference sentences (mutually exclusive with --input-tsv)
    #[arg(short, long, conflicts_with = "input_tsv")]
    pub references: Option<String>,

    /// TSV file with id, candidate, reference columns (mutually exclusive with --candidates/--references)
    #[arg(long, conflicts_with_all = ["candidates", "references"])]
    pub input_tsv: Option<String>,

    /// Output CSV file path (if not specified, prints to stdout)
    #[arg(long)]
    pub output_csv: Option<String>,

    #[clap(flatten)]
    pub tokenizer: TokenizerSpec,

    #[arg(long, default_value = "roberta")]
    pub model_type: CliEncoderModel,

    /// Model name (e.g., "roberta-large", "bert-base-uncased")
    #[arg(long, default_value = "roberta-large")]
    pub model_name: String,

    /// Use IDF weighting
    #[arg(long)]
    pub idf: bool,

    /// Use baseline rescaling
    #[arg(long)]
    pub baseline: bool,

    /// Language code for baseline rescaling (e.g., "en", "zh", "fr")
    #[arg(long, default_value = "en")]
    pub lang: String,

    /// Layer to use for scoring (default: model-specific best layer)
    #[arg(long)]
    pub layer: Option<i32>,
}

pub enum TokenizerSource {
    /// Use a pretrained tokenizer from Hugging Face
    Pretrained,
    /// Use a custom vocabulary file
    Vocab,
}

#[derive(Debug, Args)]
#[group(required = true, multiple = false, args = ["pretrained", "vocab"])]
pub struct TokenizerSpec {

    /// HF name of the pretrained tokenizer (e.g., "bert-base-uncased")
    #[arg(long)]
    pub pretrained: Option<String>,

    /// Path to vocabulary file
    #[arg(long)]
    pub vocab: Option<String>,

    /// Path to merges file (optional)
    #[arg(long)]
    pub merges: Option<String>,
}

#[derive(Args)]
pub struct SimilarityArgs {}

#[derive(Subcommand)]
pub enum Command {
    /// Score candidates against references
    Score(ScoreArgs),

    /// Compute similarity between embeddings from stdin
    Similarity(SimilarityArgs),
}
