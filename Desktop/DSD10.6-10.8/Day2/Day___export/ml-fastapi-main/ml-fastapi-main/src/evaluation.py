import logging
from sklearn.metrics import r2_score
from src.config import appconfig
from src import model_registry

logging.basicConfig(level=logging.INFO)

r2_min = float(appconfig['Evaluation']['r2'])
model_name = appconfig['Model']['name']

def get_eval_metrics(y_true, y_pred):
    """
    Return model evaluation metrics.
        Parameters:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
        Returns:
            dict: Dictionary containing evaluation metrics
    """
    results = {
        'r2': round(r2_score(y_true, y_pred),2)
    }
    return results

def run(y_true, y_pred):
    """ Main script to evaluate model performance based on R2.
        Parameters:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
        Returns:
            bool: True if evaluation passes, False otherwise
    """
    logging.info('Evaluating model...')
    new_r2 = round(r2_score(y_true, y_pred), 2)

    if new_r2 < r2_min:
        logging.warning(f"Model evaluation failed: R2 {new_r2:.2f} is below minimum required {r2_min}.")
        return False

    current_metadata = model_registry.get_metadata(model_name)
    if current_metadata is not None:
        current_r2 = current_metadata.get('metrics', {}).get('r2', None)
        if current_r2 is not None and new_r2 < current_r2:
            logging.warning(f"Model evaluation failed: R2 {new_r2:.2f} does not improve on current model R2 {current_r2:.2f}.")
            return False

    logging.info('Model evaluation passed.')
    return True
