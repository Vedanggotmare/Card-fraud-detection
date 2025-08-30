import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("creditcard.csv")
print(df.head(10))
#print(df.iloc[:,0])

print("the shape of dataframe : ", df.shape)
X = df.drop("Class", axis=1)
y = df["Class"]

# print(X)
# print(y)


#--------------------------------------------------preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X[["Time","Amount"]] = scaler.fit_transform(X[["Time","Amount"]])

#----------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

#------------------------------------------------ rebalance

from imblearn.over_sampling import SMOTE

print("before SMOTE :" , np.bincount(y_train))
sm = SMOTE(random_state=42)
X_test_res, y_train_res = sm.fit_resample(X_train,y_train)
print("after SMOTE :", np.bincount(y_train_res))

#------------------------------------------------- fitting model training

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

#logistic model
print("\ntraining the logistic regression model ...")
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_test_res,y_train_res)
print(lr)

#xgboost

print("\ntraining the xgboost model ...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,  # since we used SMOTE
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

lr.predict_proba
from tqdm import tqdm

for _ in tqdm(range(1)):
    xgb.fit(X_test_res,y_train_res)

#---------------------------------------------------evaluation and metrics
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix
import seaborn as sns

#-----------------------------------------------------defining the function for reuse
def evalutate_model(model, X_test, y_test, name = "Model"):
    y_pred = model.predict(X_test)
    y_probab = model.predict_proba(X_test)[:,1]
    print(f"\n{name} classification report : ")
    print(classification_report(y_test,y_pred=y_pred, digits= 4))
    print(f"then {name} roc_auc-score{roc_auc_score(y_test,y_probab):.4f}")

    #confusion matrix
    cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
    sns.heatmap(cm,annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} Confusion Matrix : ")
    plt.xlabel("predicated")
    plt.ylabel("actual")
    plt.show()

#---------------------------------------------------evaluating both models

evalutate_model(lr,X_test,y_test,name = "logistic")
evalutate_model(xgb,X_test,y_test,name = "Xgboost")

#----------------------------------------------------feature importance exclusive to xgboost

importances = xgb.feature_importances_
feature_name = X.columns
importances_df = pd.DataFrame({"Feature" : feature_name,"Importances" : importances})
importances_df = importances_df.sort_values(by="Importances", ascending= False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importances", y="Feature", data= importances_df)
plt.title("Importance of Feature(XGBoost)")
plt.show()


    


