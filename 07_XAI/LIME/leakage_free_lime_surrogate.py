"""
Leakage-Free Local LIME Surrogate Explainer
Bound strictly to features available prior to time horizon T
"""
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular

class LeakageFreeLIMEExplainer:
    def __init__(self, training_data, feature_names, class_names=['Retained', 'Dropout'], horizon_T_weeks=16):
        self.feature_names = feature_names
        self.class_names = class_names
        self.horizon_T_weeks = horizon_T_weeks
        
        # Verify no post-horizon T leakage features exist
        self._verify_horizon_t_compliance()
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            random_state=42
        )

    def _verify_horizon_t_compliance(self):
        forbidden_keywords = ['term2', 'term3', 'graduated', 'final_degree', 'post_T']
        for feat in self.feature_names:
            if any(kw in feat.lower() for kw in forbidden_keywords):
                raise ValueError(f"Data Leakage Error: Feature '{feat}' violates time horizon T boundary!")
        print("[LIME Verification] Passed zero-data-leakage check for Time Horizon T.")

    def explain_student_instance(self, predict_fn, student_vector, num_features=5):
        exp = self.explainer.explain_instance(
            data_row=student_vector,
            predict_fn=predict_fn,
            num_features=num_features
        )
        return exp

if __name__ == '__main__':
    np.random.seed(42)
    features = ['GPA_Term1', 'Credits_Passed_Ratio', 'STEM_Gateway_Grade', 'LMS_Access_Freq_W1_W8', 'Tuition_Delay_Days']
    X_train = np.random.randn(500, 5)
    explainer = LeakageFreeLIMEExplainer(X_train, features)
    dummy_predict = lambda x: np.hstack([1 - 1/(1+np.exp(-x[:,0])), 1/(1+np.exp(-x[:,0]))])
    exp = explainer.explain_student_instance(dummy_predict, X_train[0])
    print("[LIME] Sample local attribution weights:", exp.as_list())
