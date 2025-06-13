"""
rust-bert-score: High-performance BERTScore implementation in Rust with Python bindings.
"""

from typing import List, Tuple, Optional, Union

try:
    from . import _rust
except ImportError as e:
    raise ImportError(
        "Failed to import rust-bert-score native module. "
        "Make sure the package is properly installed with: pip install rust-bert-score"
    ) from e

__version__ = "0.2.0"

# Re-export main classes and functions
BERTScorer = _rust.BERTScorer
BERTScoreResult = _rust.PyBERTScoreResult
BaselineManager = _rust.PyBaselineManager
compute_bertscore_from_embeddings = _rust.compute_bertscore_from_embeddings


class BERTScore:
    """
    High-level API for BERTScore computation.
    
    This provides a convenient interface similar to the original bert-score package.
    """
    
    def __init__(
        self,
        model_type: str = "roberta",
        model_name: str = "roberta-large", 
        vocab_path: str,
        merges_path: Optional[str] = None,
        lang: str = "en",
        num_layers: Optional[int] = None,
        batch_size: int = 64,
        idf: bool = False,
        rescale_with_baseline: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize BERTScore.
        
        Args:
            model_type: Type of model ("bert", "roberta", "distilbert", "deberta")
            model_name: Full model name for baseline lookup
            vocab_path: Path to vocabulary file
            merges_path: Path to merges file (for BPE tokenizers)
            lang: Language code (e.g., "en", "zh")
            num_layers: Which layer to use (None for last, negative for from end)
            batch_size: Batch size for processing
            idf: Whether to use IDF weighting
            rescale_with_baseline: Whether to rescale with baseline
            device: Device to use ("cpu", "cuda", "cuda:0", etc.)
        """
        self.scorer = BERTScorer(
            model_type=model_type,
            model_name=model_name,
            vocab_path=vocab_path,
            merges_path=merges_path,
            language=lang,
            num_layers=num_layers,
            batch_size=batch_size,
            use_idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            device=device,
        )
    
    def score(
        self,
        cands: List[str],
        refs: List[str],
        return_dict: bool = False,
    ) -> Union[Tuple[List[float], List[float], List[float]], dict]:
        """
        Score candidate sentences against references.
        
        Args:
            cands: List of candidate sentences
            refs: List of reference sentences
            return_dict: If True, return dict with keys 'precision', 'recall', 'f1'
            
        Returns:
            If return_dict is False: (precision_list, recall_list, f1_list)
            If return_dict is True: {'precision': [...], 'recall': [...], 'f1': [...]}
        """
        results = self.scorer.score(cands, refs)
        
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        f1s = [r.f1 for r in results]
        
        if return_dict:
            return {
                'precision': precisions,
                'recall': recalls,
                'f1': f1s,
            }
        else:
            return precisions, recalls, f1s
    
    def score_multi_refs(
        self,
        cands: List[str],
        refs: List[List[str]],
        return_dict: bool = False,
    ) -> Union[Tuple[List[float], List[float], List[float]], dict]:
        """
        Score candidates against multiple references per candidate.
        
        Args:
            cands: List of candidate sentences
            refs: List of reference lists (one list per candidate)
            return_dict: If True, return dict with keys 'precision', 'recall', 'f1'
            
        Returns:
            Best scores (by F1) for each candidate
        """
        results = self.scorer.score_multi_refs(cands, refs)
        
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        f1s = [r.f1 for r in results]
        
        if return_dict:
            return {
                'precision': precisions,
                'recall': recalls,
                'f1': f1s,
            }
        else:
            return precisions, recalls, f1s


def score(
    cands: List[str],
    refs: List[str],
    model_type: str = "roberta",
    model_name: str = "roberta-large",
    vocab_path: str,
    merges_path: Optional[str] = None,
    lang: str = "en",
    num_layers: Optional[int] = None,
    batch_size: int = 64,
    idf: bool = False,
    rescale_with_baseline: bool = False,
    device: Optional[str] = None,
    return_dict: bool = False,
) -> Union[Tuple[List[float], List[float], List[float]], dict]:
    """
    Convenience function for one-off scoring.
    
    Creates a BERTScore instance and scores the given candidates against references.
    
    Args:
        cands: List of candidate sentences
        refs: List of reference sentences
        model_type: Type of model to use
        model_name: Full model name
        vocab_path: Path to vocabulary file
        merges_path: Path to merges file (for BPE)
        lang: Language code
        num_layers: Which layer to use
        batch_size: Batch size for processing
        idf: Whether to use IDF weighting
        rescale_with_baseline: Whether to rescale with baseline
        device: Device to use
        return_dict: If True, return dict instead of tuple
        
    Returns:
        Scores as tuple or dict depending on return_dict
    """
    scorer = BERTScore(
        model_type=model_type,
        model_name=model_name,
        vocab_path=vocab_path,
        merges_path=merges_path,
        lang=lang,
        num_layers=num_layers,
        batch_size=batch_size,
        idf=idf,
        rescale_with_baseline=rescale_with_baseline,
        device=device,
    )
    
    return scorer.score(cands, refs, return_dict=return_dict)


__all__ = [
    "BERTScore",
    "BERTScorer", 
    "BERTScoreResult",
    "BaselineManager",
    "score",
    "compute_bertscore_from_embeddings",
]