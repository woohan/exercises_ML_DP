import shap
import xgboost
import matplotlib.pyplot as plt

# get a dataset on income prediction
X,y = shap.datasets.adult()

# train an XGBoost model (but any other model type would also work)
model = xgboost.XGBClassifier()
model.fit(X, y);
# build an Exact explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Exact(model.predict_proba, X)
shap_values = explainer(X[:100])

# get just the explanations for the positive class
shap_values = shap_values[...,1]
shap.plots.bar(shap_values)
plt.savefig("./codes/XAI/imgs/test_shap.png",dpi=500, bbox_inches='tight')    

