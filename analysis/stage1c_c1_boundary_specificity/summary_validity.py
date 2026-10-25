"""A local case estimate requires two reliable source-gradient repeats."""
import numpy as np

def local_case_prediction(rows,channels):
    grouped=rows.groupby('channel')
    reliable=(grouped['index'].nunique().reindex(channels)==2)&grouped.source_gradient_valid.all().reindex(channels).fillna(False)
    p=grouped.delta_hat.mean().reindex(channels).to_numpy()
    eps=grouped.epsilon.mean().reindex(channels).to_numpy()
    magnitude=grouped.abs_gxa.mean().reindex(channels).to_numpy()
    ok=reliable.to_numpy(dtype=bool)
    p[~ok]=np.nan;eps[~ok]=np.nan;magnitude[~ok]=np.nan
    return p,eps,magnitude,ok
