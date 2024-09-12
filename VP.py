import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X_train = pd.read_csv(r"C:\Users\abhin\Downloads\training_set_features.csv")
y_train = pd.read_csv(r"C:\Users\abhin\Downloads\training_set_labels.csv")
X_test= pd.read_csv(r"C:\Users\abhin\Downloads\test_set_features.csv")


import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

most_common_category = X_train['income_poverty'].mode()[0]
X_train['income_poverty'] = X_train['income_poverty'].fillna(most_common_category)
X_test['income_poverty'] = X_test['income_poverty'].fillna(most_common_category)


age_mapping = {'55 - 64 Years': 1, '35 - 44 Years': 2, '18 - 34 Years': 3, '65+ Years': 5, '45 - 54 Years': 4}
X_train['age_group'] = X_train['age_group'].map(age_mapping)
X_test['age_group'] = X_test['age_group'].map(age_mapping)

mappings = {
    'education': {'< 12 Years': 1, '12 Years': 2, 'College Graduate': 3, 'Some College': 4},
    'census_msa': {'Non-MSA': 1, 'MSA, Not Principle City': 2, 'MSA, Principle City': 3},
    'hhs_geo_region': {'oxchjgsf': 1, 'bhuqouqj': 2, 'qufhixun': 3, 'lrircsnp': 4, 'atmpeygn': 5, 'lzgpxyit': 6, 'fpwskwrf': 7, 'mlyzmhmf': 8, 'dqpwygqj': 9, 'kbazzjca': 10},
    'employment_status': {'Not in Labor Force': 0, 'Employed': 1, 'Unemployed': -1},
    'rent_or_own': {'Own': 1, 'Rent': 0},
    'marital_status': {'Not Married': 0, 'Married': 1},
    'sex': {'Female': 0, 'Male': 1},
    'race': {'White': 1, 'Black': 2, 'Other or Multiple': 3, 'Hispanic': 4}
}

def apply_mappings(df, mappings):
    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)
    return df

X_train = apply_mappings(X_train, mappings)
X_test = apply_mappings(X_test, mappings)

income_mapping = {'<= $75,000, Above Poverty': 1, '> $75,000': 2, 'Below Poverty': 3}
X_train['income_poverty'] = X_train['income_poverty'].map(income_mapping)
X_test['income_poverty'] = X_test['income_poverty'].map(income_mapping)

X_train['doctor_recc_xyz'] = X_train['doctor_recc_xyz'].fillna(0)
X_test['doctor_recc_xyz'] = X_test['doctor_recc_xyz'].fillna(0)

X_train['doctor_recc_seasonal'] = X_train['doctor_recc_seasonal'].fillna(0)
X_train['chronic_med_condition'] = X_train['chronic_med_condition'].fillna(0)

columns_to_drop = ['health_insurance', 'employment_industry', 'employment_occupation']
X_train = X_train.drop(columns_to_drop, axis=1)
X_test = X_test.drop(columns_to_drop, axis=1)

columns_to_fill = ['marital_status', 'rent_or_own', 'education', 'employment_status', 'household_adults', 'household_children', 'census_msa']
for column in columns_to_fill:
    most_frequent_value = X_train[column].mode()[0]
    X_train[column] = X_train[column].fillna(most_frequent_value)
    X_test[column] = X_test[column].fillna(most_frequent_value)


merged_df = pd.merge(X_train, y_train, on='respondent_id')
merged_df = merged_df.dropna()
X_test = X_test.dropna()

merged_xyz = merged_df.loc[:, ~merged_df.columns.str.contains('seasonal|seas')]
X_test_xyz = X_test.loc[:, ~X_test.columns.str.contains('seasonal|seas')]

merged_seasonal = merged_df.loc[:, merged_df.columns.str.contains('seasonal|seas') | (merged_df.columns == 'respondent_id')]
X_test_sea = X_test.loc[:, X_test.columns.str.contains('seasonal|seas') | (X_test.columns == 'respondent_id')]

y_xyz = merged_xyz['xyz_vaccine']
y_sea = merged_seasonal['seasonal_vaccine']
X_xyz = merged_xyz.drop(columns=['xyz_vaccine', 'respondent_id'])
X_sea = merged_seasonal.drop(columns=['seasonal_vaccine', 'respondent_id'])
X_test_xyz = X_test_xyz.drop(columns=['respondent_id'])
X_test_sea = X_test_sea.drop(columns=['respondent_id'])


def train_and_evaluate(X_train, X_test, y_train, submission_df, label):
    lr_clf = LogisticRegression()
    lr_clf.fit(X_train, y_train)
    y_pred = lr_clf.predict(X_test)
    y_pred_proba = lr_clf.predict_proba(X_test)[:, 1]
    submission_df[label] = y_pred_proba
    return submission_df

output = pd.DataFrame(index=X_test.index)
output = train_and_evaluate(X_xyz, X_test_xyz, y_xyz, output, 'xyz_vaccine')
output = train_and_evaluate(X_sea, X_test_sea, y_sea, output, 'seasonal_vaccine')

print(output)
