import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/home/polina/.cache/kagglehub/datasets/osmi/mental-health-in-tech-2016/versions/2/mental-heath-in-tech-2016_20161114.csv")

# Remove duplicates
df = df.drop_duplicates()

# Standardize gender categories
gender_mapping = {
    "male": ["male", "m", "man", "cis male", "male.", "mail", "malr", "cis man", "cisdude", "dude", "male (cis)"],
    "female": ["female", "f", "woman", "cis female", "female/woman", "cis-woman", "fem", "fm", "fem", 
               "female (props for making this a freeform field, though)", "female assigned at birth"],
    "non-binary": ["non-binary", "bigender", "genderqueer", "genderfluid", "androgynous", "other", 
                   "genderflux demi-girl", "enby", "afab", "mtf", "genderfluid (born female)", 
                   "female-bodied; no feelings about gender", "genderqueer woman", "nb masculine", 
                   "genderqueer", "genderqueer", "genderqueer woman", "genderfluid", "genderflux demi-girl", 
                   "other/transfeminine", "genderqueer", "nonbinary", "agender"]
}

def map_gender(gender):
    gender = str(gender).lower().strip()
    for standard, options in gender_mapping.items():
        if gender in options:
            return standard
    return "other"

df['What is your gender?'] = df['What is your gender?'].apply(map_gender)

# Drop columns with more than 80% missing data
missing_data = df.isnull().mean() * 100
high_null_cols = missing_data[missing_data > 80].index
df_cleaned = df.drop(columns=high_null_cols)

# Convert Yes/No responses to binary values
binary_cols = [col for col in df_cleaned.columns if col.startswith("do_") or col.startswith("is_")]
df_cleaned[binary_cols] = df_cleaned[binary_cols].replace({"Yes": 1, "No": 0})

# Fill remaining missing values
df_cleaned = df_cleaned.fillna({
    'age': df_cleaned['age'].median(),  # Fill age with median
    'what_is_your_gender': 'other',     # Fill missing gender with 'other'
})
df_cleaned = df_cleaned.fillna("Unknown")  # Replace other NaNs with 'Unknown'

# Encoding categorical variables
df_cleaned = pd.get_dummies(df_cleaned, columns=['what_is_your_gender', 'country'], drop_first=True)

"""
# Data Analysis and Visualizations

# Age distribution of respondents
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['age'], kde=True, bins=20)
plt.title("Age Distribution of Respondents")
plt.xlabel("Age")
plt.show()

# Gender distribution
gender_counts = df['what_is_your_gender'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=gender_counts.index, y=gender_counts.values)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Distribution of mental health disorder awareness
plt.figure(figsize=(12, 8))
sns.countplot(x='do_you_currently_have_a_mental_health_disorder', data=df_cleaned)
plt.title("Current Mental Health Disorder Awareness")
plt.xlabel("Do you currently have a mental health disorder?")
plt.ylabel("Count")
plt.show()

# Correlation heatmap for numeric/binary variables
plt.figure(figsize=(15, 12))
correlation_matrix = df_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Survey Responses")
plt.show()

# Step 6: Grouped analysis (e.g., by age groups)
age_bins = [0, 18, 30, 50, 100]
age_labels = ["0-18", "19-30", "31-50", "51+"]
df_cleaned['age_group'] = pd.cut(df_cleaned['age'], bins=age_bins, labels=age_labels)

# Average rate of mental health disorder by age group
age_group_analysis = df_cleaned.groupby('age_group')['do_you_currently_have_a_mental_health_disorder'].mean()
plt.figure(figsize=(10, 6))
age_group_analysis.plot(kind='bar', color='skyblue')
plt.title("Average Rate of Mental Health Disorders by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Rate of Mental Health Disorder")
plt.show()
"""