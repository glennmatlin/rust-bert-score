//! Model loading and embedding extraction module.

use crate::Result;
use rust_bert::bert::{
    BertConfig, BertConfigResources, BertEmbeddings, BertModel, BertModelResources,
    BertVocabResources,
};
use rust_bert::deberta::{
    DebertaConfig, DebertaConfigResources, DebertaForMaskedLM, DebertaModelResources,
    DebertaVocabResources,
};
use rust_bert::distilbert::{
    DistilBertConfig, DistilBertConfigResources, DistilBertModel, DistilBertModelResources,
    DistilBertVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::roberta::RobertaForMaskedLM;
use rust_bert::roberta::{
    RobertaConfig, RobertaConfigResources, RobertaMergesResources, RobertaModelResources,
    RobertaVocabResources,
};
use rust_bert::Config;
use tch::{nn::VarStore, no_grad, Device, Tensor};

/// Supported encoder models for embedding extraction.
enum EncoderModel {
    Bert(BertModel<BertEmbeddings>),
    DistilBert(DistilBertModel),
    Roberta(RobertaForMaskedLM),
    Deberta(DebertaForMaskedLM),
}

/// Model container holding the encoder and variable store.
pub struct Model {
    _vs: VarStore,
    encoder: EncoderModel,
    device: Device,
}

struct ModelResources {
    config: Box<dyn ResourceProvider>,
    _vocab: Box<dyn ResourceProvider>,
    _merges: Option<Box<dyn ResourceProvider>>,
    weights: Box<dyn ResourceProvider>,
}

impl ModelResources {
    fn from_model_type(model_type: ModelType) -> Self {
        let config = Box::new(RemoteResource::from_pretrained(match model_type {
            ModelType::Bert => BertConfigResources::BERT,
            ModelType::DistilBert => DistilBertConfigResources::DISTIL_BERT,
            ModelType::Roberta | ModelType::XLMRoberta => RobertaConfigResources::ROBERTA,
            ModelType::Deberta => DebertaConfigResources::DEBERTA_BASE,
            _ => unimplemented!("Model type {:?} not supported", model_type),
        }));
        let vocab = Box::new(RemoteResource::from_pretrained(match model_type {
            ModelType::Bert => BertVocabResources::BERT,
            ModelType::DistilBert => DistilBertVocabResources::DISTIL_BERT,
            ModelType::Roberta | ModelType::XLMRoberta => RobertaVocabResources::ROBERTA,
            ModelType::Deberta => DebertaVocabResources::DEBERTA_BASE,
            _ => unreachable!(),
        }));
        let merges = match model_type {
            ModelType::Roberta | ModelType::XLMRoberta => Some(Box::new(
                RemoteResource::from_pretrained(RobertaMergesResources::ROBERTA),
            )
                as Box<dyn ResourceProvider>),
            _ => None,
        };
        let weights = Box::new(RemoteResource::from_pretrained(match model_type {
            ModelType::Bert => BertModelResources::BERT,
            ModelType::DistilBert => DistilBertModelResources::DISTIL_BERT,
            ModelType::Roberta | ModelType::XLMRoberta => RobertaModelResources::ROBERTA,
            ModelType::Deberta => DebertaModelResources::DEBERTA_BASE,
            _ => unimplemented!("Model type {:?} not supported", model_type),
        }));

        Self {
            config,
            _vocab: vocab,
            _merges: merges,
            weights,
        }
    }
}

impl Model {
    /// Load a pre-trained encoder (BERT, DistilBERT, or RoBERTa) configured to output hidden states.
    pub fn new(model_type: ModelType, device: Device) -> Result<Self> {
        // Select resources
        let resources = ModelResources::from_model_type(model_type);

        let mut var_store = VarStore::new(device);

        let model_path = resources.config.get_local_path()?;
        let weights_path = resources.weights.get_local_path()?;

        let encoder = match model_type {
            ModelType::Bert => {
                let mut config = BertConfig::from_file(model_path);
                config.output_hidden_states = Some(true);
                let model = BertModel::<BertEmbeddings>::new(var_store.root(), &config);
                var_store.load(weights_path)?;
                EncoderModel::Bert(model)
            }
            ModelType::DistilBert => {
                let mut config = DistilBertConfig::from_file(model_path);
                config.output_hidden_states = Some(true);
                let model = DistilBertModel::new(var_store.root(), &config);
                var_store.load(weights_path)?;
                EncoderModel::DistilBert(model)
            }
            ModelType::Roberta | ModelType::XLMRoberta => {
                let mut config = RobertaConfig::from_file(model_path);
                config.output_hidden_states = Some(true);
                let model = RobertaForMaskedLM::new(var_store.root(), &config);
                var_store.load(weights_path)?;
                EncoderModel::Roberta(model)
            }
            ModelType::Deberta => {
                let mut config = DebertaConfig::from_file(model_path);
                config.output_hidden_states = Some(true);
                let model = DebertaForMaskedLM::new(var_store.root(), &config);
                var_store.load(weights_path)?;
                EncoderModel::Deberta(model)
            }
            _ => unimplemented!("Model type {:?} not supported", model_type),
        };

        Ok(Model {
            _vs: var_store,
            encoder,
            device,
        })
    }

    /// Forward pass: produces all hidden-state tensors for each layer (including embeddings).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: Option<&Tensor>,
    ) -> Vec<Tensor> {
        let input_ids = input_ids.to_device(self.device);
        let attention_mask = attention_mask.to_device(self.device);
        let token_type_ids = token_type_ids.map(|t| t.to_device(self.device));

        no_grad(|| match &self.encoder {
            EncoderModel::Bert(model) => {
                model.forward_hidden_states(input_ids, attention_mask, token_type_ids)
            }
            EncoderModel::DistilBert(model) => {
                model.forward_hidden_states(input_ids, attention_mask, token_type_ids)
            }
            EncoderModel::Roberta(model) => {
                model.forward_hidden_states(input_ids, attention_mask, token_type_ids)
            }
            EncoderModel::Deberta(model) => {
                model.forward_hidden_states(input_ids, attention_mask, token_type_ids)
            }
        })
    }
}

// Model Trait for extracting hidden states

trait ForwardHiddenStates {
    fn forward_hidden_states(
        &self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Option<Tensor>,
    ) -> Vec<Tensor>;
}

impl ForwardHiddenStates for BertModel<BertEmbeddings> {
    fn forward_hidden_states(
        &self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Option<Tensor>,
    ) -> Vec<Tensor> {
        self.forward_t(
            Some(&input_ids),
            Some(&attention_mask),
            token_type_ids.as_ref(),
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap()
        .all_hidden_states
        .unwrap()
    }
}

impl ForwardHiddenStates for DistilBertModel {
    fn forward_hidden_states(
        &self,
        input_ids: Tensor,
        attention_mask: Tensor,
        _token_type_ids: Option<Tensor>,
    ) -> Vec<Tensor> {
        self.forward_t(Some(&input_ids), Some(&attention_mask), None, false)
            .unwrap()
            .all_hidden_states
            .unwrap()
    }
}

impl ForwardHiddenStates for RobertaForMaskedLM {
    fn forward_hidden_states(
        &self,
        input_ids: Tensor,
        attention_mask: Tensor,
        _token_type_ids: Option<Tensor>,
    ) -> Vec<Tensor> {
        self.forward_t(
            Some(&input_ids),
            Some(&attention_mask),
            None,
            None,
            None,
            None,
            None,
            false,
        )
        .all_hidden_states
        .unwrap()
    }
}

impl ForwardHiddenStates for DebertaForMaskedLM {
    fn forward_hidden_states(
        &self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Option<Tensor>,
    ) -> Vec<Tensor> {
        self.forward_t(
            Some(&input_ids),
            Some(&attention_mask),
            token_type_ids.as_ref(),
            None,
            None,
            false,
        )
        .unwrap()
        .all_hidden_states
        .unwrap()
    }
}
