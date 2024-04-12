import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Predictive Modeling in Political Voting 2024")
st.markdown("Indian Election Prediction involves analyzing various factors such as demographic data, historical voting patterns, public opinion surveys, and socio-economic indicators to forecast the outcome of elections in India. With the vast diversity in the electorate and the complex political landscape, accurate prediction requires sophisticated statistical models and machine learning algorithms. Factors such as party alliances, candidate popularity, campaign strategies, and regional dynamics also play crucial roles. Election prediction in India is not only a matter of statistical analysis but also involves understanding the pulse of the electorate and interpreting the ever-changing political scenario to provide meaningful insights and forecasts.")
df=pd.read_csv(r"https://raw.githubusercontent.com/KunalKumbhar13/Election-prediction-2024/main/Streamlit/election%20survey%202_csv.csv")
st.write("Raw Data:")
st.dataframe(df)
st.write("Columns:",df.shape[1])
st.write("Rows:",df.shape[0])
df.info()
df.isnull().sum()

##male female disribution
df['Gender:\n'].value_counts().plot(kind='bar', color='skyblue', legend=False)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
st.subheader('Male and Female Distribution Ratio :woman::man:', divider='rainbow')
st.subheader('Male Ratio is Higher as compare to Female.')
st.pyplot(plt.show())

##voting pattern by genders
sns.countplot(x='Gender:\n', data=df, hue=' Will you vote in upcoming elections?\n')
st.subheader('Voting Pattern by Gender:woman::man:', divider='rainbow')
st.subheader('Male Count is Higher as compare to Female.')
st.pyplot(plt.show())


##GENDER ENCODING

df = pd.get_dummies(df, columns=['Gender:\n'], prefix='Gender:\n', drop_first=True)
df['Gender:\n_male'] = df['Gender:\n_male'].astype(int)
print(df.columns)



##age encoding
##age distribution
age_counts = df[' Age:\n'].value_counts().sort_index()
age_counts.plot(kind='bar', color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
st.subheader('Age Distribution:girl::boy::woman::man::white_haired_man:', divider='rainbow')
st.subheader('As we can see Age Group between 21 to 55 have Highest count. ')
st.pyplot(plt.show())



age_mapping = {
    '21 to 30': 1,
    '31 to 45': 2,
    '46 to 60': 3,
    '61 to 75': 4,
    '75 and above': 5
}

df['Age'] = df[' Age:\n'].map(age_mapping)

df.drop(columns=[' Age:\n'], inplace=True)

##education encoding



df[' Educational Qualification:\n'].unique()


education_mapping = {
    'Non-Literate': 0,
    'Primary School': 1,
    'Secondary School/Matric': 2,
    'Higher Secondary/Intermediate': 3,
    'Graduate': 4,
    'Postgraduate': 5
}

# Map the educational qualifications to numerical values
df['Educational Qualification'] = df[' Educational Qualification:\n'].map(education_mapping)

df.drop(columns=[' Educational Qualification:\n'], inplace=True)

##occupation encoding


df[' Occupation:\n'].unique()

df = pd.get_dummies(df, columns=[' Occupation:\n'], prefix=' Occupation:\n')

columns_to_convert = [' Occupation:\n_Employed (Govt.)', ' Occupation:\n_Farmer',
       ' Occupation:\n_Homemaker', ' Occupation:\n_Other',
       ' Occupation:\n_Retired', ' Occupation:\n_Self-employed',
       ' Occupation:\n_Student', ' Occupation:\n_Unemployed']

df[columns_to_convert] = df[columns_to_convert].replace({True: 1, False: 0})


##religion distribution
religion_counts = df['religion \n'].value_counts()

# Define custom colors for each religion
colors = {
    'Hindu': 'orange',
    'Islam': 'green',
    'Buddhist': 'blue',
    'Christian': 'red',
    'No Response':'purple',
    'Other':'brown'
}

# Create the pie chart with custom colors
plt.figure(figsize=(8, 8))
plt.pie(religion_counts, labels=religion_counts.index, autopct='%1.1f%%', colors=[colors[religion] for religion in religion_counts.index])
plt.title('Distribution of Religion')
st.subheader('Distribution By Religion 🕉️☪️☸️✝️', divider='rainbow')
st.subheader("As we can see Hindu Population is greater as compare to Other Religion's")
st.pyplot(plt.show())

df.rename(columns={'religion \n': 'Religion'}, inplace=True)

df = pd.get_dummies(df, columns=['Religion'], prefix='Religion')

df['Religion_No Response'].sum()

df = df[df['Religion_No Response'] == False]

df.drop(columns=['Religion_No Response'],inplace=True)
df.head()

df['Area/locality\n'].isna().sum()

df.dropna(subset=['Area/locality\n'], inplace=True)

df.rename(columns={'Area/locality\n': 'Locality'}, inplace=True)

df = pd.get_dummies(df, columns=['Locality'], prefix='Locality')

##most imp election issues
df[' most important election issue for you?'].value_counts().plot(kind='bar', color='red', legend=False)
plt.title('MIMP election issue')
plt.xlabel('issues')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
st.subheader('Important Issues In India:heavy_exclamation_mark:', divider='rainbow')
st.subheader("Price Rise/Inflation is the BIG ISSUE faced by our Country:chart_with_downwards_trend:")
st.pyplot(plt.show())

st.subheader("Distribution of votes to re_elect_govt by Occupation:male-scientist::judge::male-scientist::male-technologist:",divider='rainbow')
st.subheader("Young People and Govermnet (Emp) have High counts for votes:thumbsup:")
st.image("G:\Python\Streamlit\output4.png", caption='Every One Should Vote', use_column_width=True)

#df.columns

columns_to_convert = ['Religion_Buddhist', 'Religion_Christian', 'Religion_Hindu',
       'Religion_Islam', 'Religion_Other',
       'Locality_Rural/Village', 'Locality_Urban/Big city',
       'Locality_Urban/Small city']

df[columns_to_convert] = df[columns_to_convert].replace({True: 1, False: 0})

## imp election issues or personality
df['issues vs person?'].value_counts().plot(kind='bar', color='green', legend=False)
plt.title('issue or person')
#plt.xlabel('issues')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
st.subheader('Issues Or Political Personality:question:', divider='rainbow')
st.subheader("Election Issues has High Frequency.:chart_with_upwards_trend:")
st.pyplot(plt.show())
##encoding mothly income
df.rename(columns={' monthly household income \n': 'Monthly Income'}, inplace=True)

from sklearn.preprocessing import OrdinalEncoder

# Define the order of categories based on their ordinal relationship
income_order = ['Over Rs. 50001', 'Rs. 20,000 to Rs. 50,000',
       'Rs. 10,001 - Rs. 20,000', 'Up to Rs.10,000']

# Initialize OrdinalEncoder with specified order
ordinal_encoder = OrdinalEncoder(categories=[income_order])

# Apply ordinal encoding to the 'monthly household income' column
df['Income'] = ordinal_encoder.fit_transform(df[['Monthly Income']])

df.drop(columns=['Monthly Income'],inplace=True)

df.head()
df['will vote']=df[' Will you vote in upcoming elections?\n']

df.drop(columns=[' Will you vote in upcoming elections?\n'],inplace=True)

df['will vote'] = df['will vote'].replace({'Yes': 1, 'No': 0})

df.columns=['trad_supp',
       'current_strong_pos',
       'your_vote_for',
       'imp_issue', 'issues vs personality?',
       'Voting Priorities', 'MP Satisfaction?',
       'MP Performance Rating', 'current Govt. Satisfaction',
       're_elect govt', 'Gender_male', 'Age', 'Educational Qualification',
       ' Occupation_Employed (Govt.)', ' Occupation_Farmer',
       ' Occupation_Homemaker', ' Occupation_Other',
       ' Occupation_Retired', ' Occupation_Self-employed',
       ' Occupation_Student', ' Occupation_Unemployed','Religion_Buddhist', 'Religion_Christian', 'Religion_Hindu',
       'Religion_Islam', 'Religion_Other', 'Locality_Rural/Village',
       'Locality_Urban/Big city', 'Locality_Urban/Small city', 'Income',
       'will vote']


#df.columns

#df.shape[0]

##traditionally supported encoding
df['trad_supp'].unique()

df=df[df['trad_supp'] != 'No party']

df=pd.get_dummies(df,columns=['trad_supp'])

map={
    'trad_supp_Aam Aadmi Party':'trad_AAP',
    'trad_supp_BJP':'trad_BJP'	,
    'trad_supp_Congress':'trad_congress',
    'trad_supp_party from Indian National Developmental Inclusive Alliance (INDIA)':'trad_india',
    'trad_supp_party from National Democratic Alliance (ND)':'trad_nda'
}

df=df.rename(columns=map)

##curent strong
df['current_strong_pos'].unique()

df=pd.get_dummies(df,columns=['current_strong_pos'])

map2={
    'current_strong_pos_Aam Aadmi Party':'strg_aap',
    'current_strong_pos_BJP':'strg_bjp',
    'current_strong_pos_Congress':'strg_congress',
    'current_strong_pos_Indian National Developmental Inclusive Alliance (INDIA)':'strg_india',
    'current_strong_pos_National Democratic Alliance (NDA)':'strg_nda'
}

df=df.rename(columns=map2)

##making copy of df for "your vote" targeting
df_vote=df.copy()

##your vote encoding
df['your_vote_for'].unique()

df=pd.get_dummies(df,columns=['your_vote_for'])

map3={
    'your_vote_for_Aam Aadmi Party':'vote_aap',
    'your_vote_for_BJP':'vote_bjp',
    'your_vote_for_Congress':'vote_congress',
    'your_vote_for_party from Indian National Developmental Inclusive Alliance (INDIA)':'vote_india',
    'your_vote_for_party from National Democratic Alliance (ND)':'vote_nda'

}

df=df.rename(columns=map3)

##isuues vs personality encoding


df['issues vs personality?'].unique()

df=pd.get_dummies(df,columns=['issues vs personality?'])

map4={
    'issues vs personality?_Election issues':'electn_isuues',
    'issues vs personality?_Other':'other',
    "issues vs personality?_political leader's face/personality":"leader"

}

df=df.rename(columns=map4)




##encoding mps satisfaction
df['MP Satisfaction?'].unique()

m = len(df[df['MP Satisfaction?'] == 'No Response'])
#m

most_frequent_value = df['MP Satisfaction?'].mode()[0]
entry_to_replace = 'No Response'
df.loc[df['MP Satisfaction?'] == entry_to_replace, 'MP Satisfaction?'] = most_frequent_value

df['MP Satisfaction?'] = df['MP Satisfaction?'].replace({'Yes': 1, 'No': 0})

##encoding 'MP Performance Rating'
df['MP Performance Rating'].unique()

m = len(df[df['MP Performance Rating'] == 'No Response'])
#m

most_frequent_value = df['MP Performance Rating'].mode()[0]
entry_to_replace = 'No Response'
df.loc[df['MP Performance Rating'] == entry_to_replace, 'MP Performance Rating'] = most_frequent_value

ordinal_mapping = {
    'Very poor': 1,
    'Poor': 2,
    'Satisfactory': 3,
    'Good': 4,
    'Excellent': 5
}

df['MP Performance Rating'] = df['MP Performance Rating'].map(ordinal_mapping)

df.head()

##encoding current gov
df['current Govt. Satisfaction'].unique()

df['current Govt. Satisfaction'] = df['current Govt. Satisfaction'].replace({'Yes': 1, 'No': 0})


##encoding reelct govt


df['re_elect govt'].unique()

df['re_elect govt'] = df['re_elect govt'].replace({'Yes': 1, 'No': 0})

df = df.replace({True: 1, False: 0})


df2=df.drop(columns=['imp_issue','Voting Priorities'])
df2.head()

df2.isnull().sum()

column_mode = df2['Age'].mode()[0]

df2['Age'].fillna(column_mode, inplace=True)
#now no null values

st.subheader("Count of Tradictionly Supported and Currently their strong choice",divider='rainbow')
st.subheader("As we can see BJP Previously also have strong public Support and Current Support also.:handshake:")
st.image("G:\Python\Streamlit\output2.png", caption='Strong Support of People towards BJP', use_column_width=True)
st.subheader("Current Govt. Satisfaction vs. Re-elect govt:question:",divider='rainbow')
st.subheader("As we can see People want Change.:arrows_counterclockwise:")
st.image("G:\Python\Streamlit\output3.png", caption='Winning Part BJP.', use_column_width=True)
st.subheader("Preference of Vote if Elections are held Tommorow	:grey_question:",divider='rainbow')
st.subheader("As we can see BJP has Winning Luck in this Election.:first_place_medal:")
st.image("G:\Python\Streamlit\output1.png", caption='Winning Party BJP.', use_column_width=True)


# ##Spliting Data
# X = df2.drop(columns=['re_elect govt'])
# y = df2['re_elect govt']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ##Random Forest
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)
# st.subheader("Using Random Forest Algorithm for Training and Testing Data:-",divider="rainbow")
# # Model evaluation
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# st.subheader("Acurracy using Random Forest:", divider='rainbow')
# st.success(accuracy, icon=None)


# from sklearn.metrics import confusion_matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(conf_matrix)
# st.subheader("Confusion Matrix:", divider='rainbow')
# st.success("""Confusion Matrix:
# [[21  0]
        
#  [ 0 50]]""")


# ##Logistic

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# ##logistic
# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# st.subheader("Using Logistic Regression for Training and Testing Data:-",divider="rainbow")
# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# st.subheader("Acurracy using Logistic Regression:", divider='rainbow')
# st.success(accuracy, icon=None)
# # Generate a classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# st.subheader("Classification Report:", divider='rainbow')
# st.success("""
#            precision    recall  f1-score   support

#            0       1.00      1.00      1.00        21
#            1       1.00      1.00      1.00        50

#     accuracy                           1.00        71
#    macro avg       1.00      1.00      1.00        71
# weighted avg       1.00      1.00      1.00        71
# """)


# ##Cross Validation

# from sklearn.model_selection import cross_val_score

# # cv=5 means 5-fold cross-validation
# # scoring='accuracy' specifies the metric to be used for evaluation
# scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# # Print the cross-validation scores
# print("Cross-Validation Scores:", scores)
# st.subheader("Cross-Validation Scores:", divider='rainbow')
# st.success(scores, icon=None)



# # Calculate and print the mean and standard deviation of the scores
# print("Mean Accuracy:", scores.mean())
# print("Standard Deviation:", scores.std())
# st.subheader("Mean Accuracy:", divider='rainbow')
# st.success(scores.mean(), icon=None)
# st.subheader("Standard Deviation:", divider='rainbow')
# st.success(scores.std(), icon=None)



df_vote=df_vote.drop(columns=['imp_issue','Voting Priorities'])
df_vote=pd.get_dummies(df_vote,columns=['issues vs personality?'])

map4={
    'issues vs personality?_Election issues':'electn_isuues',
    'issues vs personality?_Other':'other',
    "issues vs personality?_political leader's face/personality":"leader"

}

df_vote=df_vote.rename(columns=map4)

most_frequent_value = df_vote['MP Satisfaction?'].mode()[0]
entry_to_replace = 'No Response'
df_vote.loc[df_vote['MP Satisfaction?'] == entry_to_replace, 'MP Satisfaction?'] = most_frequent_value

df_vote['MP Satisfaction?'] = df_vote['MP Satisfaction?'].replace({'Yes': 1, 'No': 0})

most_frequent_value = df_vote['MP Performance Rating'].mode()[0]
entry_to_replace = 'No Response'
df_vote.loc[df_vote['MP Performance Rating'] == entry_to_replace, 'MP Performance Rating'] = most_frequent_value

ordinal_mapping = {
    'Very poor': 1,
    'Poor': 2,
    'Satisfactory': 3,
    'Good': 4,
    'Excellent': 5
}

df_vote['MP Performance Rating'] = df_vote['MP Performance Rating'].map(ordinal_mapping)

df_vote['current Govt. Satisfaction'] = df_vote['current Govt. Satisfaction'].replace({'Yes': 1, 'No': 0})

df_vote['re_elect govt'] = df_vote['re_elect govt'].replace({'Yes': 1, 'No': 0})

df_vote = df_vote.replace({True: 1, False: 0})
df_vote.isnull().sum()

df_vote['Age'].fillna(df['Age'].mode()[0],inplace=True)

df_vote['your_vote_for'].unique()

le = LabelEncoder()
df_vote['your_vote_for'] = le.fit_transform(df_vote['your_vote_for'])

st.subheader("Problem Statements",divider="rainbow")
st.write("""1. Using a machine learning model, predict voting behavior of people, i.e. which party they 
will vote for based on demographic, socioeconomic, and political variables.
2. Using a machine learning model, predict whether people would vote for the governing 
party again based on demographic, socioeconomic, and political variables.""")

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np

X1, y1 = df_vote.drop(columns=['your_vote_for']), df_vote['your_vote_for']

# Split data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Select top k features using chi-squared test
k = 10  # Number of top features to select
selector = SelectKBest(score_func=chi2, k=k)
X1_train_selected = selector.fit_transform(X1_train, y1_train)
X1_test_selected = selector.transform(X1_test)

# Get the indices of the selected features
selected_indices = np.where(selector.get_support())[0]

# Print the indices of selected features
print("Indices of selected features:", selected_indices)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming X1_train_selected and X1_test_selected are the selected features obtained after feature selection
# Assuming y1_train and y1_test are the corresponding target labels

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier on the selected features
rf_classifier.fit(X1_train_selected, y1_train)

# Predict on the test set
y1_pred = rf_classifier.predict(X1_test_selected)

# Evaluate accuracy
accuracy1 = accuracy_score(y1_test, y1_pred)
st.subheader("Using Random Forest Algorithm for Training and Testing Data:-",divider="rainbow")
print("Accuracy1:", accuracy1)
st.subheader("Acurracy using Random Forest for Problem Statement 1:", divider='rainbow')
st.success(accuracy1, icon=None)

from sklearn.metrics import classification_report

# Assuming X1_train_selected and X1_test_selected are the selected features obtained after feature selection
# Assuming y1_train and y1_test are the corresponding target labels

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier on the selected features
rf_classifier.fit(X1_train_selected, y1_train)

# Predict on the test set
y1_pred = rf_classifier.predict(X1_test_selected)

# Generate classification report
report = classification_report(y1_test, y1_pred)
st.subheader("Classification Report:",divider="rainbow")
st.success(report)

X2 = df2.drop(columns=['re_elect govt'])
y2 = df2['re_elect govt']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Select top k features using chi-squared test
k = 10  # Number of top features to select
selector = SelectKBest(score_func=chi2, k=k)
X2_train_selected = selector.fit_transform(X2_train, y2_train)
X2_test_selected = selector.transform(X2_test)

# Get the indices of the selected features
selected_indices = np.where(selector.get_support())[0]

# Print the indices of selected features
print("Indices of selected features:", selected_indices)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming X2_train_selected and X2_test_selected are the selected features obtained after feature selection
# Assuming y2_train and y2_test are the corresponding target labels

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier on the selected features
rf_classifier.fit(X2_train_selected, y2_train)

# Predict on the test set
y2_pred = rf_classifier.predict(X2_test_selected)

# Evaluate accuracy
accuracy2 = accuracy_score(y2_test, y2_pred)
print("Accuracy2:", accuracy2)
st.subheader("Acurracy using Random Forest for Problem Statement 2:", divider='rainbow')
st.success(accuracy2, icon=None)


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y2_test, y2_pred)
print("Confusion Matrix:")
print(conf_matrix)
st.subheader("Confusion Matrix:", divider='rainbow')
st.success(conf_matrix, icon=None)

# Assuming X2_train_selected and X2_test_selected are the selected features obtained after feature selection
# Assuming y2_train and y2_test are the corresponding target labels

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier on the selected features
rf_classifier.fit(X2_train_selected, y2_train)

# Predict on the test set
y2_pred = rf_classifier.predict(X2_test_selected)

# Generate classification report
report = classification_report(y2_test, y2_pred)
print(report)
st.subheader("Classification Report:",divider="rainbow")
st.success(report)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
##logistic
model = LogisticRegression()
model.fit(X2_train, y2_train)
y3_pred = model.predict(X2_test)

# Calculate accuracy
accuracy3 = accuracy_score(y2_test, y2_pred)
st.subheader("Using Logistic Regression for Training and Testing Data:-",divider="rainbow")
st.subheader("Acurracy using Logistic Regression for Problem Statement 2:", divider='rainbow')
print("Accuracy3:", accuracy3)
st.success(accuracy3, icon=None)


# Generate a classification report
print("Classification Report:")
print(classification_report(y2_test, y2_pred))
st.subheader("Classification Report Using Logistic Regression:", divider='rainbow')
st.success((classification_report(y2_test, y2_pred)), icon=None)

from sklearn.model_selection import cross_val_score

# cv=5 means 5-fold cross-validation
# scoring='accuracy' specifies the metric to be used for evaluation
scores = cross_val_score(model, X2, y2, cv=5, scoring='accuracy')

# Print the cross-validation scores
st.subheader("Cross-Validation Scores:", divider='rainbow')
print("Cross-Validation Scores:", scores)
st.success(scores,icon=None)

# Calculate and print the mean and standard deviation of the scores
st.subheader("Mean Accuracy:", divider='rainbow')
print("Mean Accuracy:", scores.mean())
st.success(scores.mean(),icon=None)
st.subheader("Standard Deviation:", divider='rainbow')
print("Standard Deviation:", scores.std())
st.success(scores.std(),icon=None)


