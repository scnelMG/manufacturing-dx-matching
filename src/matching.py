"""FAISS retrieval and provider ranking used by notebook 06."""

import faiss
import numpy as np
import pandas as pd


def retrieve_technology_candidates(
    demand_embedding: np.ndarray,
    technology_embeddings: np.ndarray,
    technology_metadata: pd.DataFrame,
    top_k: int = 100,
) -> pd.DataFrame:
    if technology_embeddings.ndim != 2:
        raise ValueError('technology_embeddings must be a two-dimensional array.')
    if demand_embedding.ndim != 1:
        raise ValueError('demand_embedding must be a one-dimensional array.')
    if technology_embeddings.shape[1] != demand_embedding.shape[0]:
        raise ValueError('Demand and technology embedding dimensions must match.')
    if len(technology_embeddings) != len(technology_metadata):
        raise ValueError('Embedding and metadata lengths must match.')
    if not 1 <= top_k <= len(technology_embeddings):
        raise ValueError('top_k must be between 1 and the number of technologies.')

    technology_vectors = technology_embeddings.astype('float32').copy()
    demand_vector = demand_embedding.reshape(1, -1).astype('float32').copy()
    faiss.normalize_L2(technology_vectors)
    faiss.normalize_L2(demand_vector)

    index = faiss.IndexFlatIP(technology_vectors.shape[1])
    index.add(technology_vectors)
    similarities, indices = index.search(demand_vector, top_k)

    candidates = technology_metadata.iloc[indices[0]].copy()
    candidates['similarity'] = similarities[0]
    return candidates.reset_index(drop=True)

def rank_provider_candidates(
    candidate_df: pd.DataFrame,
    provider_evidence_df: pd.DataFrame,
    technology_weight: float = 0.8,
) -> pd.DataFrame:
    if not 0 <= technology_weight <= 1:
        raise ValueError('technology_weight must be between 0 and 1.')

    required_candidate_columns = {'provider_company', 'technology_name', 'similarity'}
    required_evidence_columns = {'provider_company', 'award_count', 'certificate_count'}
    if missing := required_candidate_columns - set(candidate_df.columns):
        raise KeyError(f'Missing candidate columns: {sorted(missing)}')
    if missing := required_evidence_columns - set(provider_evidence_df.columns):
        raise KeyError(f'Missing evidence columns: {sorted(missing)}')
    if candidate_df['similarity'].isna().any():
        raise ValueError('similarity must not contain missing values.')
    if provider_evidence_df['provider_company'].duplicated().any():
        raise ValueError('provider_evidence_df must contain one row per provider_company.')
    if (provider_evidence_df[['award_count', 'certificate_count']] < 0).any().any():
        raise ValueError('award_count and certificate_count must be non-negative.')

    provider_df = candidate_df.groupby('provider_company', as_index=False).agg(
        avg_similarity=('similarity', 'mean'),
        matched_technology_count=('technology_name', 'count'),
    )
    provider_df['relevance_score'] = (
        provider_df['avg_similarity'] * np.log1p(provider_df['matched_technology_count'])
    )
    result_df = provider_df.merge(provider_evidence_df, on='provider_company', how='left').fillna({
        'award_count': 0,
        'certificate_count': 0,
    })
    result_df['credibility_score'] = np.log1p(
        result_df['award_count'] + result_df['certificate_count']
    )

    def min_max_scale(series: pd.Series) -> pd.Series:
        value_range = series.max() - series.min()
        if value_range == 0:
            return pd.Series(0.0, index=series.index)
        return (series - series.min()) / value_range * 100

    result_df['relevance_score_norm'] = min_max_scale(result_df['relevance_score'])
    result_df['credibility_score_norm'] = min_max_scale(result_df['credibility_score'])
    result_df['final_score'] = (
        technology_weight * result_df['relevance_score_norm'] +
        (1 - technology_weight) * result_df['credibility_score_norm']
    )
    return result_df.sort_values('final_score', ascending=False).reset_index(drop=True)
