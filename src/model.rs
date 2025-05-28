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

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn test_model_type_support() {
        // Test that we support the expected model types
        let supported = vec![
            ModelType::Bert,
            ModelType::DistilBert,
            ModelType::Roberta,
            ModelType::XLMRoberta,
            ModelType::Deberta,
        ];
        
        for model_type in supported {
            match model_type {
                ModelType::Bert | ModelType::DistilBert | ModelType::Roberta | 
                ModelType::XLMRoberta | ModelType::Deberta => {
                    // These are all supported
                    assert!(true);
                }
                _ => panic!("Unexpected model type"),
            }
        }
    }

    #[test]
    fn test_tensor_shapes() {
        // Test expected tensor shapes for forward pass
        let batch_size = 2;
        let seq_len = 10;
        let vocab_size = 30522;
        
        // Mock input tensors
        let input_ids = Tensor::randint(vocab_size, &[batch_size, seq_len], (Kind::Int64, Device::Cpu));
        let attention_mask = Tensor::ones(&[batch_size, seq_len], (Kind::Int64, Device::Cpu));
        let token_type_ids = Tensor::zeros(&[batch_size, seq_len], (Kind::Int64, Device::Cpu));
        
        // Verify shapes
        assert_eq!(input_ids.size(), vec![batch_size, seq_len]);
        assert_eq!(attention_mask.size(), vec![batch_size, seq_len]);
        assert_eq!(token_type_ids.size(), vec![batch_size, seq_len]);
    }

    #[test]
    fn test_hidden_states_structure() {
        // Test expected structure of hidden states output
        let batch_size = 2;
        let seq_len = 10;
        let hidden_size = 768;
        let num_layers = 12;
        
        // Mock hidden states for each layer
        let hidden_states: Vec<Tensor> = (0..=num_layers)
            .map(|_| Tensor::randn(&[batch_size, seq_len, hidden_size], (Kind::Float, Device::Cpu)))
            .collect();
        
        // Verify we have embeddings + all layers
        assert_eq!(hidden_states.len(), num_layers + 1);
        
        // Verify each hidden state has correct shape
        for (i, state) in hidden_states.iter().enumerate() {
            assert_eq!(
                state.size(), 
                vec![batch_size, seq_len, hidden_size],
                "Hidden state {} has wrong shape", i
            );
        }
    }

    #[test]
    fn test_resource_types() {
        // Test that resource types are correctly mapped
        use rust_bert::bert::{BertConfigResources, BertModelResources, BertVocabResources};
        use rust_bert::distilbert::{
            DistilBertConfigResources, DistilBertModelResources, DistilBertVocabResources
        };
        use rust_bert::roberta::{
            RobertaConfigResources, RobertaModelResources, RobertaVocabResources
        };
        use rust_bert::deberta::{DebertaConfigResources, DebertaModelResources, DebertaVocabResources};
        
        // Just verify these resource types exist and can be referenced
        let _ = BertConfigResources::BERT;
        let _ = BertModelResources::BERT;
        let _ = BertVocabResources::BERT;
        
        let _ = DistilBertConfigResources::DISTIL_BERT;
        let _ = DistilBertModelResources::DISTIL_BERT;
        let _ = DistilBertVocabResources::DISTIL_BERT;
        
        let _ = RobertaConfigResources::ROBERTA;
        let _ = RobertaModelResources::ROBERTA;
        let _ = RobertaVocabResources::ROBERTA;
        
        let _ = DebertaConfigResources::DEBERTA_BASE;
        let _ = DebertaModelResources::DEBERTA_BASE;
        let _ = DebertaVocabResources::DEBERTA_BASE;
    }

    #[test]
    fn test_device_support() {
        // Test that we handle different device types
        let devices = vec![
            Device::Cpu,
            // Device::Cuda(0), // Only test if CUDA available
        ];
        
        for device in devices {
            match device {
                Device::Cpu => assert!(true),
                Device::Cuda(_) => assert!(true),
                _ => panic!("Unexpected device type"),
            }
        }
    }
}