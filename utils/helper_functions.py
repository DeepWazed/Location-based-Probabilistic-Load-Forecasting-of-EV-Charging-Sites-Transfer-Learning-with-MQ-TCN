from fastai.tabular.model import emb_sz_rule


def get_approach(model_type):
    if model_type == 0:
        return "TL_BM"
    else:
        return "TL_TM"

def get_emb_sz_list(dims: list):
    """
    For all elements in the given list, find a size for the respective embedding through trial and error
    Each element denotes the amount of unique values for one categorical feature
    Parameters
    ----------
    dims : list
        a list containing a number of integers.
    Returns
    -------
    list of tupels
        a list containing an the amount of unique values and respective embedding size for all elements.
    """
    return [(d, emb_sz_rule(d)) for d in dims]