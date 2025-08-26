import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

def csv_setup(d='loan_approval_dataset.csv'):
    df=pd.read_csv(d).head(500)
    
    if 'Application_ID' in df.columns:
        df.drop('Application_ID', axis=1, inplace=True)
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)
    if 'Previous_Details' in df.columns:
        df.drop('Previous_Details', axis=1, inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    
    return df
    
def splitting_test(d):
    X = d[['Age', 'Income', 'Credit_Score', 'Loan_Amount',
           'Loan_Term', 'Interest_Rate', 'Employment_Status',
           'Debt_to_Income_Ratio', 'Marital_Status',
           'Number_of_Dependents', 'Property_Ownership',
           'Loan_Purpose']]
    y=d['Previous_Defaults']
    le=LabelEncoder()

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X.loc[:,col] = le.fit_transform(X[col].astype(str))


    return X,y

def Training_test(X,y):
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    smote=SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    final_testing={}
    testing_algorithm={'Logistic regression':LogisticRegression(max_iter=1000),
                       'SVM':SVC(),
                       'Decision Tree':DecisionTreeClassifier(),
                       'Knn':KNeighborsClassifier()}
    
    for model_name,models in testing_algorithm.items():
        models.fit(X_train,y_train)
        y_pred=models.predict(X_test)
        acc=accuracy_score(y_test,y_pred)
        final_testing[model_name]=acc

        print("\nModel:\n",model_name)
        print("\nThe Accuracy Test for loan is\n",acc)
        print("\nThe Confusion Matrix fro loan is\n",confusion_matrix(y_test,y_pred))
        print("\nThe Classification result for loan is\n",classification_report(y_test,y_pred))

    return final_testing

def plotting(final_testing):
    plt.figure(figsize=(8,5))
    plt.plot(
        list(final_testing.keys()),   
        list(final_testing.values()), 
        marker='o', linestyle='-', color='b'
    )
    plt.title("Loan Accuracy Prediction")
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.grid()
    plt.show()

def end():
    df=csv_setup()
    X,y=splitting_test(df)
    final_testing=Training_test(X,y)
    plotting(final_testing)

end()
