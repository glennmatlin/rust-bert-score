use std::path::PathBuf;

use hf_hub::api::sync::Api;

pub fn fetch_vocab_files(
    model_name: &str,
) -> Result<(PathBuf, Option<PathBuf>), anyhow::Error> {
    let api = Api::new()?;

    let repo = api.model(model_name.to_string());

    // Try to fetch vocab.txt, if not found, try vocab.json
    let vocab_path = repo.get("vocab.txt").or_else(|_| repo.get("vocab.json")).map_err(
        |e| anyhow::anyhow!("Failed to fetch either vocab.txt or vocab.json: {}", e)
    )?;

    let merges_path = repo.get("merges.txt").ok();

    Ok((vocab_path, merges_path))

}