//! Model loading and embedding extraction module.

use crate::Result;
use rust_bert::pipelines::common::ModelType;
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::bert::{
    BertConfig, BertConfigResources, BertEmbeddings, BertModel, BertModelResources,
    BertVocabResources,
};
use rust_bert::distilbert::{
    DistilBertConfig, DistilBertConfigResources, DistilBertModel,
    DistilBertModelResources, DistilBertVocabResources,
};
use rust_bert::roberta::{
    RobertaConfig, RobertaConfigResources, RobertaMergesResources,
    RobertaModelResources, RobertaVocabResources,
};
use rust_bert::roberta::RobertaForMaskedLM;
use rust_bert::Config;
use rust_bert::deberta::{
    DebertaConfig, DebertaConfigResources, DebertaForMaskedLM,
    DebertaModelResources, DebertaVocabResources,
};
use tch::{no_grad, nn::VarStore, Device, Tensor};

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
    _device: Device,
}

impl Model {
    /// Load a pre-trained encoder (BERT, DistilBERT, or RoBERTa) configured to output hidden states.
    pub fn new(model_type: ModelType, device: Device) -> Result<Self> {
        // Select resources
        let config_res = Box::new(RemoteResource::from_pretrained(match model_type {
            ModelType::Bert => BertConfigResources::BERT,
            ModelType::DistilBert => DistilBertConfigResources::DISTIL_BERT,
            ModelType::Roberta | ModelType::XLMRoberta => RobertaConfigResources::ROBERTA,
            ModelType::Deberta => DebertaConfigResources::DEBERTA_BASE,
            _ => unimplemented!("Model type {:?} not supported", model_type),
        }));
        let _vocab_res = Box::new(RemoteResource::from_pretrained(match model_type {
            ModelType::Bert => BertVocabResources::BERT,
            ModelType::DistilBert => DistilBertVocabResources::DISTIL_BERT,
            ModelType::Roberta | ModelType::XLMRoberta => RobertaVocabResources::ROBERTA,
            ModelType::Deberta => DebertaVocabResources::DEBERTA_BASE,
            _ => unreachable!(),
        }));
        let _merges_res = match model_type {
            ModelType::Roberta | ModelType::XLMRoberta => Some(Box::new(
                RemoteResource::from_pretrained(RobertaMergesResources::ROBERTA),
            )),
            _ => None,
        };
        let weights_res = Box::new(RemoteResource::from_pretrained(match model_type {
            ModelType::Bert => BertModelResources::BERT,
            ModelType::DistilBert => DistilBertModelResources::DISTIL_BERT,
            ModelType::Roberta | ModelType::XLMRoberta => RobertaModelResources::ROBERTA,
            ModelType::Deberta => DebertaModelResources::DEBERTA_BASE,
            _ => unimplemented!("Model type {:?} not supported", model_type),
        }));

        // Initialize var store and encoder based on model type
        let mut vs = VarStore::new(device);
        let encoder = match model_type {
            ModelType::Bert => {
                let path = config_res.get_local_path()?;
                let mut config = BertConfig::from_file(path);
                config.output_hidden_states = Some(true);
                let model = BertModel::<BertEmbeddings>::new(&vs.root(), &config);
                let weights = weights_res.get_local_path()?;
                vs.load(weights)?;
                EncoderModel::Bert(model)
            }
            ModelType::DistilBert => {
                let path = config_res.get_local_path()?;
                let mut config = DistilBertConfig::from_file(path);
                config.output_hidden_states = Some(true);
                let model = DistilBertModel::new(&vs.root(), &config);
                let weights = weights_res.get_local_path()?;
                vs.load(weights)?;
                EncoderModel::DistilBert(model)
            }
            ModelType::Roberta | ModelType::XLMRoberta => {
                let path = config_res.get_local_path()?;
                let mut config = RobertaConfig::from_file(path);
                config.output_hidden_states = Some(true);
                let model = RobertaForMaskedLM::new(&vs.root(), &config);
                let weights = weights_res.get_local_path()?;
                vs.load(weights)?;
                EncoderModel::Roberta(model)
            }
            ModelType::Deberta => {
                let path = config_res.get_local_path()?;
                let mut config = DebertaConfig::from_file(path);
                config.output_hidden_states = Some(true);
                let model = DebertaForMaskedLM::new(&vs.root(), &config);
                let weights = weights_res.get_local_path()?;
                vs.load(weights)?;
                EncoderModel::Deberta(model)
            }
            _ => unimplemented!("Model type {:?} not supported", model_type),
        };
        Ok(Model { _vs: vs, encoder, _device: device })
    }

    /// Forward pass: produces all hidden-state tensors for each layer (including embeddings).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: Option<&Tensor>,
    ) -> Vec<Tensor> {
        no_grad(|| match &self.encoder {
            EncoderModel::Bert(model) => {
                let output = model
                    .forward_t(
                        Some(input_ids),
                        Some(attention_mask),
                        token_type_ids,
                        None,
                        None,
                        None,
                        None,
                        false,
                    )
                    .unwrap();
                output.all_hidden_states.unwrap()
            }
            EncoderModel::DistilBert(model) => {
                let output = model.forward_t(Some(input_ids), Some(attention_mask), None, false)
                    .unwrap();
                output.all_hidden_states.unwrap()
            }
            EncoderModel::Roberta(model) => {
                let output = model.forward_t(
                    Some(input_ids),
                    Some(attention_mask),
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                );
                output.all_hidden_states.unwrap()
            }
            EncoderModel::Deberta(model) => {
                let output = model
                    .forward_t(
                        Some(input_ids),
                        Some(attention_mask),
                        token_type_ids,
                        None,
                        None,
                        false,
                    )
                    .unwrap();
                output.all_hidden_states.unwrap()
            }
        })
    }
}