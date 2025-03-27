from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class PatchedXGBClassifier(XGBClassifier, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __sklearn_tags__(self):
        return {
            "requires_y": True,
            "X_types": ["2darray"],
            "allow_nan": True,
            "estimator_type": "classifier"
        }
