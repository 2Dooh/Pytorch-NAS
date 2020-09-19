import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./data/loan_prediction/train.csv')

gender = {'Male': 1, 'Female': 0}
married = {'Yes': 1, 'No': 0}
dependents = {'0': 0, '1': 1, '3+': 2}
education = {'Graduate': 1, 'Not Graduate': 0}
self_employed = {'Yes': 1, 'No': 0}
property_area = {'Rural': 0, 'Urban': 2, 'Semiurban': 1}
loan_status = {'Y': 0, 'N': 1}


new_sample = dataset.drop(['Loan_ID'], axis=1)

new_sample['Gender'] = new_sample['Gender'].map(gender)
new_sample['Married'] = new_sample['Married'].map(married)
new_sample['Dependents'] = new_sample['Dependents'].map(dependents)
new_sample['Education'] = new_sample['Education'].map(education)
new_sample['Self_Employed'] = new_sample['Self_Employed'].map(self_employed)
new_sample['Property_Area'] = new_sample['Property_Area'].map(property_area)
new_sample['Loan_Status'] = new_sample['Loan_Status'].map(loan_status)

new_sample = new_sample.dropna()


new_sample['ApplicantIncome'] = new_sample['CoapplicantIncome'] / new_sample['ApplicantIncome']
new_sample['LoanAmount'] = new_sample['LoanAmount'] * new_sample['Loan_Amount_Term']


new_sample.drop(['CoapplicantIncome', 'Loan_Amount_Term'], axis=1, inplace=True)

# mu = new_sample.iloc[:, :-1].mean(axis=0)
# sigma = new_sample.iloc[:, :-1].std(axis=0)   

# new_sample.iloc[:, :-1] = (new_sample.iloc[:, :-1] - mu) / sigma
corr = new_sample.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True)
plt.show()

