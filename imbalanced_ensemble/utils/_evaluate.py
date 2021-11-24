from sklearn.metrics import balanced_accuracy_score, f1_score
from imbalanced_ensemble.metrics import geometric_mean_score

DEFAULT_METRICS = {
    'balanced Acc': (balanced_accuracy_score, {}),
    'macro Fscore': (f1_score, {'average':'macro'}),
    'macro Gmean': (geometric_mean_score, {'average':'macro'}),
}

def evaluate_print(y_true, y_pred, head:str="", 
                   eval_metrics:dict=DEFAULT_METRICS, 
                   print_str:bool=True, return_str:bool=False):
    """Evaluate and print the predictive performance with respect to 
    the given metrics.

    Returns a string of evaluation results.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    head : string, default=""
        Head of the returned string, for example, the name of the predictor.

    eval_metrics : dict, default=None
        Metric(s) used for evaluation during the ensemble training process.

        - If ``None``, use 3 default metrics:

            - ``'balanced Acc'``: 
                ``sklearn.metrics.balanced_accuracy_score()``
            - ``'macro F1'``: 
                ``sklearn.metrics.f1_score(average='macro')``
            - ``'macro Gmean'``: 
                ``imbens.metrics.geometric_mean_score(average='macro')``

        - If ``dict``, the keys should be strings corresponding to evaluation 
            metrics' names. The values should be tuples corresponding to the metric 
            function (``callable``) and additional kwargs (``dict``).

            - The metric function should at least take 2 named/keyword arguments, 
                ``y_true`` and one of [``y_pred``, ``y_score``], and returns a float
                as the evaluation score. Keyword arguments:

                - ``y_true``, 1d-array of shape (n_samples,), true labels or binary 
                label indicators corresponds to ground truth (correct) labels.
                - When using ``y_pred``, input will be 1d-array of shape (n_samples,) 
                corresponds to predicted labels, as returned by a classifier.
                - When using ``y_score``, input will be 2d-array of shape (n_samples, 
                n_classes,) corresponds to probability estimates provided by the 
                predict_proba method. In addition, the order of the class scores 
                must correspond to the order of ``labels``, if provided in the metric 
                function, or else to the numerical or lexicographical order of the 
                labels in ``y_true``.
            
            - The metric additional kwargs should be a dictionary that specifies 
                the additional arguments that need to be passed into the metric function. 

    print_str : bool, defaul=True
        Whether to print the results to stdout. If False, disable print.

    return_str : bool, defaul=False
        Whether to return the result string. If True, returns it.
        
    Returns
    -------
    result_str : string or NoneType

    """
    result_str = head + " "
    for metric_name, (metric_func, kwargs) in eval_metrics.items():
        score = metric_func(y_true, y_pred, **kwargs)
        result_str += "{}: {:.3f} | ".format(metric_name, score)
    if print_str:
        print (result_str.rstrip(" |"))
    if return_str:
        return result_str.rstrip(" |")