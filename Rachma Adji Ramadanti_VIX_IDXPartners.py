
#Credit Risk Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix

#Import Data
dataset = pd.read_csv("C:/Users/User/Downloads/loan_data_2007_2014.csv")
#Cek informasi dari data
dataset.info()

# Preprocessing Data - Exploratory Data Analysis

# Data Understanding

# Cek Data Duplikat
#Melihat apakah seluruh baris dalam dataset ada yang duplikat atau tidak
if dataset[dataset.duplicated()].empty:
    print("Tidak ada baris duplikat dalam DataFrame.")
else:
    print("Baris Duplikat:")
    print(dataset[dataset.duplicated()])

# Cek Data Hilang (Missing Value)

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
cek_missing = missing_data(dataset)
cek_missing

# Feature Engineering
# Pengkategorian Fitur
# Mengkategorikan Negara Berdasarkan Region 
#Dictionary untuk wilayah di AS
us_region_dict = {
    'Northeast': ['CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT'],
    'Midwest': ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'],
    'South': ['AL', 'AR', 'DE', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV'],
    'West': ['AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']
}
dataset['Region'] = dataset['addr_state'].apply(lambda x: next((region for region, countries in us_region_dict.items() if x in countries), None))
dataset['Region']

# Fungsi untuk mengklasifikasi int rate 
def convert_int_rate(int_rate):
    if int_rate < 13.66 :
        return 'Low'
    else:
        return 'High'

# Mengaplikasikan fungsi ke dalam DataFrame
dataset['int_rate_int'] = dataset['int_rate'].apply(convert_int_rate)

#Klasifikasi Income Peminjam
def classify_income(income):
    if income < 100000:
        return 'Low'
    elif 100000 <= income < 200000:
        return 'Medium'
    else:
        return 'High'

dataset['Klasifikasi_Pendapatan'] = dataset['annual_inc'].apply(classify_income)

# Fungsi untuk mengonversi employment_length ke dalam tipe data integer
def convert_employment_length(employment_length):
    if pd.isna(employment_length) or employment_length == 'n/a':
        return None
    elif '< 1 year' in employment_length:
        return 0
    elif '1 year' in employment_length:
        return 1
    elif '10+ years' in employment_length:
        return 10
    else:
        # Ambil angka dari string (contoh: '2 years' menjadi 2)
        return int(''.join(filter(str.isdigit, employment_length)))

# Mengaplikasikan fungsi ke dalam DataFrame
dataset['emp_length_int'] = dataset['emp_length'].apply(convert_employment_length)


#Konversi term dari string ke dalam tipe data numerik
dataset['term'] = dataset['term'].map({' 36 months': 36, ' 60 months': 60})
dataset['term'].unique()

# Konversi loan status kedalam 2 kategori yaitu Good Loan = 0 dan Bad Loan = 1
dataset['loan_status'].unique()
def encode_with_custom_labels(column, custom_labels):
    encoded_column = column.map(custom_labels)
    return encoded_column
keterangan = {
    'Fully Paid':0, 'Charged Off':1, 'Current':0, 'Default':1,
       'Late (31-120 days)':1, 'In Grace Period':1, 'Late (16-30 days)':1,
       'Does not meet the credit policy. Status:Fully Paid':0,
       'Does not meet the credit policy. Status:Charged Off':1 }
dataset['loan_condition'] = encode_with_custom_labels(dataset['loan_status'],keterangan)

# Fungsi untuk mengubah format tanggal
def convert_date_format(date_str):
    # Periksa jika nilai adalah string sebelum melakukan parsing
    if isinstance(date_str, str):
        date_object = datetime.strptime(date_str, "%b-%y")
        return date_object.strftime("%Y")
    else:
        return None
dataset['issue_d'] = dataset['issue_d'].apply(convert_date_format)
dataset['last_pymnt_d'] = dataset['last_pymnt_d'].apply(convert_date_format)
dataset['next_pymnt_d'] = dataset['next_pymnt_d'].apply(convert_date_format)
dataset['last_credit_pull_d'] = dataset['last_credit_pull_d'].apply(convert_date_format)
dataset['earliest_cr_line'] = dataset['earliest_cr_line'].apply(convert_date_format)


# Visualisasi
loan_counts = dataset.groupby(['Region', 'loan_status']).size().unstack()
# Plot bar chart
fig, ax = plt.subplots(figsize=(10, 6))
loan_counts.plot(kind='bar', stacked=True, ax=ax)

# Menambahkan label dan judul
plt.xlabel('Region')
plt.ylabel('Number of Loans')
plt.title('Loan Status Composition by Region')

# Menambahkan legenda
plt.legend(title='Loan Status', bbox_to_anchor=(1.05, 1), loc='upper left')

# Menampilkan plot
plt.show()

# Membuat boxplot menggunakan Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Klasifikasi_Pendapatan', y='loan_amnt', data=dataset, palette='Set3')
plt.title('Plotting Peminjam Berdasarkan Pinjaman')
plt.xlabel('Klasifikasi Pendapatan')
plt.ylabel('Loan Amount')

# Menampilkan plot
plt.show()

# Membuat boxplot menggunakan Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Klasifikasi_Pendapatan', y='int_rate', data=dataset, palette='Set3')
plt.title('Plotting Peminjam Berdasarkan Suku Bunga Pinjaman')
plt.xlabel('Klasifikasi Pendapatan')
plt.ylabel('Interest Rate')

# Menampilkan plot
plt.show()

# Plot bar chart
# Membuat bar chart menggunakan Seaborn
plt.figure(figsize=(10, 6))
sns.countplot(x='Klasifikasi_Pendapatan', hue='emp_length_int', data=dataset, palette='viridis')
plt.title('Bar Chart Lama Kerja pada Tiap Klasifikasi Pendapatan')
plt.xlabel('Klasifikasi Pendapatan')
plt.ylabel('Jumlah Peminjam')
# Menampilkan plot
plt.show()

# Plot bar chart
# Membuat bar chart menggunakan Seaborn
plt.figure(figsize=(10, 6))
sns.countplot(x='Klasifikasi_Pendapatan', hue='loan_condition', data=dataset, palette='viridis')
plt.title('Bar Chart Kondisi Loan pada Tiap Klasifikasi Pendapatan')
plt.xlabel('Klasifikasi Pendapatan')
plt.ylabel('Jumlah Peminjam')
plt.legend(title='Loan Condition', loc='upper right', labels=['Good', 'Bad'])
# Menampilkan plot
plt.show()

# Plot bar chart
# Membuat bar chart menggunakan Seaborn
plt.figure(figsize=(10, 6))
sns.countplot(x='purpose', hue='loan_condition', data=dataset, palette='viridis')
plt.title('Alasan Pinjaman Berdasarkan Status Pinjaman')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Klasifikasi Pendapatan')
plt.ylabel('Jumlah Peminjam')
plt.legend(title='Loan Condition', loc='upper right', labels=['Good', 'Bad'])
# Menampilkan plot
plt.show()

# Membuat catplot menggunakan Seaborn
plt.figure(figsize=(10, 6))

sns.countplot(x='issue_d', hue='loan_condition', data=dataset,
            palette={0: 'green', 1: 'red'})

# Menambahkan judul dan label sumbu
plt.title('Perbandingan Kondisi Pinjaman Tiap Tahun')
plt.xlabel('Tahun Pemberian Pinjaman')
plt.ylabel('Jumlah Peminjam')
plt.legend(title='Loan Condition', loc='upper left', labels=['Good', 'Bad'])

# Menampilkan plot
plt.show()

# Misalkan 'loan_status' adalah nama kolom kondisi pinjaman dalam DataFrame Anda
loan_status_counts = dataset['loan_condition'].value_counts()

# Membuat pie chart
plt.figure(figsize=(8, 8))
plt.pie(loan_status_counts, labels=['Good Loan', 'Bad Loan'], autopct='%1.1f%%', startangle=140)
plt.title('Distribusi Kondisi Pinjaman')
plt.show()

# Feature Selection

# Menghapus fitur yang tidak digunakan dan sudah dikategorikan sebelumnya
column_1 = [dataset.columns[0],'id','member_id','emp_title','url','purpose','desc','zip_code','title','loan_status','emp_length','annual_inc','addr_state','issue_d', 'earliest_cr_line', 'last_pymnt_d','last_credit_pull_d']
dataset = dataset.drop(columns = column_1)
dataset.info()

### Hapus Kolom yang memiliki banyak missing value
def remove_columns_with_high_missing_values(df, threshold):
    # Hitung total nilai yang hilang untuk setiap kolom
    missing_values = df.isnull().sum()

    # Urutkan kolom berdasarkan total nilai yang hilang secara menurun
    sorted_columns = missing_values.sort_values(ascending=False)

    # Pilih kolom-kolom dengan total nilai yang hilang tinggi
    high_missing_columns = sorted_columns[sorted_columns > threshold].index

    # Hapus kolom-kolom tersebut dari DataFrame
    df_filtered = df.drop(columns=high_missing_columns)

    return df_filtered

dataset = remove_columns_with_high_missing_values(dataset,0.1*len(dataset))
dataset.info()

#Menghilangkan baris yang terdapat missing value
dataset.dropna(inplace=True)
missing_data(dataset)


# Procesing Data

#Mempersiapkan data untuk modeling menggunakan Machine Learning
#Labeling fitur dengan tipe data string
label_encoder = LabelEncoder()
dataset['grade'] = label_encoder.fit_transform(dataset['grade'])
dataset['sub_grade'] = label_encoder.fit_transform(dataset['sub_grade'])
dataset['home_ownership'] = label_encoder.fit_transform(dataset['home_ownership'])
dataset['verification_status'] = label_encoder.fit_transform(dataset['verification_status'])
dataset['pymnt_plan'] = label_encoder.fit_transform(dataset['pymnt_plan'])
dataset['initial_list_status'] = label_encoder.fit_transform(dataset['initial_list_status'])
dataset['application_type'] = label_encoder.fit_transform(dataset['application_type'])
dataset['Region'] = label_encoder.fit_transform(dataset['Region'])
dataset['Klasifikasi_Pendapatan'] = label_encoder.fit_transform(dataset['Klasifikasi_Pendapatan'])
dataset['int_rate_int'] = label_encoder.fit_transform(dataset['int_rate_int'])
dataset.info()


# Splitting Data

X = dataset.drop(['loan_condition'], axis=1)
y = dataset['loan_condition']

stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Melakukan split dan mendapatkan indeks untuk training set dan testing set
for train_index, test_index in stratified_splitter.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# Handling Imbalance - SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)  # 'auto' menyesuaikan jumlah sampel dengan kelas minoritas
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(pd.Series(y_resampled).value_counts())


# Data Modeling

# Building Model Training

# Fungsi untuk melatih dan mengevaluasi model Logistic Regression
def train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test, max_iter):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Logistic Regression')
    plt.legend(loc='lower right')
    plt.show()
    hasil = {
        'Model' : 'Logistic Regression' ,
        'y_pred': y_pred,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'AUC ROC': roc_auc
    }

    return hasil
# Fungsi untuk melatih dan mengevaluasi model XGBoost
def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, n_estimators):
    model = XGBClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve XGBoost')
    plt.legend(loc='lower right')
    plt.show()
    hasil = {
        'Model' : 'XGBoost',
        'y_pred': y_pred,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'AUC ROC': roc_auc
    }

    return hasil


#Evaluation Model

#Evaluasi pada data training
# Inisialisasi nilai yang akan diuji
max_iter_values = [50, 100, 200]
n_estimators_values_xgb = [50, 100, 200]

# List untuk menyimpan hasil eksperimen
results = []

# Lakukan eksperimen untuk Logistic Regression
for max_iter in max_iter_values:
    results.append(train_and_evaluate_logistic_regression(X_resampled, y_resampled, X_resampled, y_resampled, max_iter))

# Lakukan eksperimen untuk XGBoost
for n_estimators in n_estimators_values_xgb:
    results.append(train_and_evaluate_xgboost(X_resampled, y_resampled, X_resampled, y_resampled, n_estimators))

# Buat DataFrame dari hasil eksperimen
results_df = pd.DataFrame(results)

# Tampilkan DataFrame
print(results_df)


#Evaluasi pada data testing
# Inisialisasi nilai yang akan diuji
max_iter_values = [50, 100, 200]
n_estimators_values_xgb = [50, 100, 200]

# List untuk menyimpan hasil eksperimen
results = []

# Lakukan eksperimen untuk Logistic Regression
for max_iter in max_iter_values:
    results.append(train_and_evaluate_logistic_regression(X_resampled, y_resampled, X_test, y_test, max_iter))

# Lakukan eksperimen untuk XGBoost
for n_estimators in n_estimators_values_xgb:
    results.append(train_and_evaluate_xgboost(X_resampled, y_resampled, X_test, y_test, n_estimators))

# Buat DataFrame dari hasil eksperimen
results_df = pd.DataFrame(results)

# Tampilkan DataFrame
print(results_df)

# Prediksi
result = train_and_evaluate_xgboost(X_resampled, y_resampled, X_test, y_test, n_estimators = 200)
# Mengakses y_pred, accuracy, dan f1_score
predictions = result['y_pred']
accuracy = result['Accuracy']
f1 = result['F1-score']
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)    
    
# Plot confusion matrix dengan seaborn
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap="Blues")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
print("Hasil Prediksi:")
print(list(predictions))
print("Akurasi:", accuracy)
print("F1-score:", f1)

# Menghitung jumlah setiap kategori
counts = {status: list(predictions).count(status) for status in set(list(predictions))}
print(counts)

# Membuat plot
plt.figure(figsize=(8, 6))
plt.bar(['Good','Bad'], counts.values(), color=['green', 'red'])  # Warna hijau untuk 'Approved' dan merah untuk 'Rejected'
plt.title('Count Plot for Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()
