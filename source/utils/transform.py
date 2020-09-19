import torch

class LoanPredictionToNumeric(object):
    def __call__(self, sample):
        gender = {'Male': 1, 'Female': 0}
        married = {'Yes': 1, 'No': 0}
        dependents = {'0': 0, '1': 1, '3+': 2}
        education = {'Graduate': 1, 'Not Graduate': 0}
        self_employed = {'Yes': 1, 'No': 0}
        property_area = {'Rural': 0, 'Urban': 2, 'Semiurban': 1}
        loan_status = {'Y': 1, 'N': 0}


        transformed_sample = sample.drop(['Loan_ID'], axis=1)
        transformed_sample['Gender'] = sample['Gender'].map(gender)
        transformed_sample['Married'] = sample['Married'].map(married)
        transformed_sample['Dependents'] = sample['Dependents'].map(dependents)
        transformed_sample['Education'] = sample['Education'].map(education)
        transformed_sample['Self_Employed'] = sample['Self_Employed'].map(self_employed)
        transformed_sample['Property_Area'] = sample['Property_Area'].map(property_area)
        if 'Loan_Status' in transformed_sample.columns:
            transformed_sample['Loan_Status'] = sample['Loan_Status'].map(loan_status)

        transformed_sample = transformed_sample.fillna(-1)

        return transformed_sample

class FeatureNormalizer(object):
    def __call__(self, sample):
        transformed_sample = sample.copy()
        mu = sample.iloc[:, :-1].mean(axis=0)
        sigma = sample.iloc[:, :-1].max(axis=0) - sample.iloc[:, :-1].min(axis=0)
        transformed_sample.iloc[:, :-1] = (sample - mu) / sigma

        return transformed_sample.to_numpy()

class Squeeze(object):
    def __call__(self, sample):
        return torch.squeeze(sample).type(torch.float)