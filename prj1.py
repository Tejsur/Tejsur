# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:44:46 2023

@author: shruti
"""




import pandas as pd #add library
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

import json
from io import StringIO




file_path = r"C:\Users\shruti\Desktop\sql project\transactions.txt"

data_list = []

with open(file_path, 'r') as file: 
    for line in file:
        try:
            data_dict = json.loads(line.strip())
            data_list.append(data_dict)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

df = pd.DataFrame(data_list) 
df

df.describe()

d1=df.corr() 
corr=df[["creditLimit","availableMoney","transactionAmount","currentBalance","isFraud"]].corr()
sns.heatmap(corr,annot=True)

sns.heatmap(df.corr())

df = df.replace(r'', np.NaN)#replace blank value with NaN 
df.isnull().sum()

 #drop Nan columns
df.drop(['echoBuffer', 'merchantCity', 'merchantState', 'merchantZip', 'posOnPremises', 'recurringAuthInd'], axis=1, inplace=True)
df



df.customerId.nunique()   #unique number of customers
df.merchantName.nunique()  #unique number of merchants
df.accountNumber.nunique() #unique number of account numbers



df.transactionType.value_counts() 
df.acqCountry.value_counts()
df.merchantCategoryCode.value_counts()





 

x1=df.transactionType.value_counts()
df



df.loc[df.isFraud==True]
df.loc[df.isFraud==False]



fraud_by_type = df.groupby('transactionType')['isFraud'].mean() * 100
print("Percentage of Fraudulent Transactions by Type:","\n",fraud_by_type) 


country_summary = df.groupby(['acqCountry', 'merchantCountryCode']).size().unstack(fill_value=0)
print("Geographic Analysis:")
print(country_summary)

country_summary = df.groupby(['acqCountry', 'merchantCountryCode']).agg({'isFraud': 'sum','transactionAmount': 'count'}).unstack(fill_value=0)
print("Geographic Analysis (Including Fraud Counts):")
cs=print(country_summary)


import seaborn as sns
import matplotlib.pyplot as plt

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(country_summary['isFraud'], annot=True, fmt='d', cmap='YlGnBu')
plt.title('Count of Fraudulent Transactions by Geographic Locations')
plt.xlabel('Merchant Country Code')
plt.ylabel('Acquisition Country')
plt.show()







fraud_trxn = df[df['isFraud']==True]
len(fraud_trxn)
non_fraud_trxn = df[df['isFraud'] == False]
len(non_fraud_trxn)


#bar Graph of top 10 fraudulent account
plt.figure(figsize=(15,6))
fraud_accounts_count = fraud_trxn['accountNumber'].value_counts()
fraud_accounts_count.head(10).sort_values().plot(kind = 'barh', color='red')
plt.xlabel('Count of Fraud Transactions', fontsize=15)
plt.ylabel('Account Number', fontsize=15)
plt.title('Top 10 Accounts with High Fraudulent Transactions', fontsize=16)



#histogram of Transaction amount
plt.figure(figsize=(14,6))
sns.histplot(data=df, x="transactionAmount", color = 'blue', bins="auto")
plt.title('Histogram of Transaction Amount', fontsize=16)



plt.figure(figsize=(8,5))
sns.histplot(data=df,x="transactionAmount",kde=True,color="green",bins="auto")


plt.figure(figsize=(8,6))
sns.countplot(x="isFraud",data=df,palette="Set2")

plt.figure(figsize=(8,6))
sns.barplot(data=df,x=df.isFraud.value_counts(),y=["True","False"],palette="Set2")
plt.title("Bar Chart of isFraud")
plt.xlabel("Count",fontsize=14)
plt.ylabel("isFraud",fontsize=14)

plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='transactionAmount', bins=30, kde=True, color='skyblue')
plt.title('Distribution of Transaction Amount', fontsize=17)
plt.xlabel('Transaction Amount', fontsize=14)
plt.ylabel('Frequency', fontsize=14)


plt.figure(figsize=(10,6))
sns.barplot(data=df, x='transactionType', y='transactionAmount', hue='isFraud')
plt.title('Average Transaction Amount by Type and Fraud Status')


plt.figure(figsize=(10,6))
sns.barplot(data=df, x='transactionType', y='transactionAmount', hue='isFraud')
plt.title('Average Transaction Amount by Type and Fraud Status')



# Add values on the bars



average_transaction_amount = df.groupby(['transactionType', 'isFraud'])['transactionAmount'].mean().unstack(fill_value=0)
print(average_transaction_amount)




plt.figure(figsize=(8,6))
sns.histplot(data=df, x="creditLimit", color = 'red', bins="auto")
plt.title("Histogram of CreditLimit", fontsize=16)

plt.figure(figsize=(8,6))
sns.histplot(data=df, x="availableMoney", color = 'blue', bins="auto")
plt.title("Histogram of availableMoney", fontsize=16)


# bar plot for % of Fraud transactions
plt.figure(figsize=(8,6))
sns.barplot(data=df,x=fraud_by_type.index,y=fraud_by_type.values)
plt.title("Percentage of Fraudulent Transactions by Type", fontsize=16)
plt.xlabel("Transaction Type",fontsize=14)
plt.ylabel("% of Fraudulent Transactions",fontsize=14)


#Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='creditLimit', y='transactionAmount', hue='isFraud')
plt.title('Scatter Plot of Credit Limit vs. Transaction Amount by Fraud Status')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='creditLimit', y='transactionAmount')
plt.title('Scatter Plot of Credit Limit vs. Transaction Amount by Fraud Status')


#### ------ Merchanat analysis

# Top merchants by transaction count
top_merchants_count = df['merchantName'].value_counts().head(10)

# Top merchants by transaction amount
top_merchants_amount = df.groupby('merchantName')['transactionAmount'].sum().nlargest(10)

print("Top Merchants by Transaction Count:")
print(top_merchants_count)
print("\nTop Merchants by Transaction Amount:")
print(top_merchants_amount)


# fraudulent transactions for each merchant
fraud_proportion_by_merchant = df.groupby('merchantName')['isFraud'].mean().sort_values(ascending=False)

# Select merchants with a higher proportion of fraudulent transactions
fraudulent_merchants = fraud_proportion_by_merchant[fraud_proportion_by_merchant > 0]

print("Merchants with a Higher Proportion of Fraudulent Transactions:")
print(fraudulent_merchants)


plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='merchantCategoryCode', y='transactionAmount')
plt.title('Distribution of Transaction Amounts by Merchant Category')


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, y='creditLimit', orient='vertical', color='lightblue')
plt.xlabel('Credit Limit')
plt.ylabel('Value')
plt.title('Boxplot of Credit Limit')

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, y='transactionAmount', orient='vertical', color='green')


plt.figure(figsize=(8,6))
sns.countplot(data=df, x='customerId', hue='isFraud', palette={False: 'blue', True: 'red'})
plt.title('Card Usage Frequency (Fraud vs. Non-Fraud)')
plt.xlabel('Customer ID')
plt.ylabel('Frequency')
plt.legend(['Non-Fraud', 'Fraud'])


plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='customerId', hue='isFraud', element='step', common_norm=False, palette={False: 'blue', True: 'red'})
plt.title('Card Usage Frequency (Fraud vs. Non-Fraud)')
plt.xlabel('Customer ID')
plt.ylabel('Frequency')
plt.legend(['Non-Fraud', 'Fraud']) 


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='customerId', y=None, hue='isFraud', palette={False: 'blue', True: 'red'}, s=50, alpha=0.6)
plt.title('Card Usage Frequency (Fraud vs. Non-Fraud)')
plt.xlabel('Customer ID')
plt.ylabel('Frequency')
plt.legend(['Non-Fraud', 'Fraud'])



fraud_by_merchant_category = df.groupby('merchantCategoryCode')['isFraud'].mean() * 100
print(fraud_by_merchant_category)

plt.figure(figsize=(12, 6))
sns.barplot(x=fraud_by_merchant_category.index, y=fraud_by_merchant_category.values)
plt.title('Fraud Percentage by Merchant Category')
plt.xlabel('Merchant Category')
plt.ylabel('Percentage of Fraudulent Transactions') 


# Group transactions by customer and count the number of transactions
customer_transaction_counts = df.groupby('customerId')['transactionAmount'].count()
customer_transaction_counts

plt.figure(figsize=(10, 6))
sns.histplot(customer_transaction_counts, bins=30, kde=True)
plt.title('Transaction Frequency Distribution')
plt.xlabel('Number of Transactions')
plt.ylabel('Frequency')


non_fraud_customer_transaction_counts = non_fraud_trxn.groupby('customerId')['transactionAmount'].count()
plt.figure(figsize=(10, 6))
sns.histplot(non_fraud_customer_transaction_counts, bins=30, kde=True, color='blue', label='Non-Fraud')
plt.title('Transaction Frequency Distribution (Non-Fraudulent)')
plt.xlabel('Number of Transactions')
plt.ylabel('Frequency')




fraud_counts = df.groupby(['transactionType', 'isFraud'])['transactionType'].count().unstack(fill_value=0)
print("Transaction Type vs. Fraud Counts:")
print(fraud_counts)



fraud_counts1 = df[df['transactionType'] == 'ADDRESS_VERIFICATION']['isFraud'].value_counts()


    
    
    
    
fraud_by_card = df.groupby('customerId')['isFraud'].mean()*100
print("Percentage of Fraudulent Transactions by Cardholder:")
print(fraud_by_card.head(10))
    

# Create a new column 'cvv_verification_result' to store the result
df['cvv_verification_result'] = 'CVV Mismatched'  # Initialize with a default value

# Check if 'cardCVV' matches 'enteredCVV' and update 'cvv_verification_result' accordingly
df.loc[df['cardCVV'] == df['enteredCVV'], 'cvv_verification_result'] = 'CVV Matched'

df

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='cvv_verification_result', palette={"CVV Matched": "green", "CVV Mismatched": "red"})
plt.title('CVV Verification Result')
plt.xlabel('Verification Result')
plt.ylabel('Count')



########## time series analysis 
 
# Convert the 'transactionDateTime' column to a datetime data type
df['transactionDateTime'] = pd.to_datetime(df['transactionDateTime']) 

# Set the 'transactionDateTime' column as the DataFrame's index (for time series analysis)
df.set_index('transactionDateTime', inplace=True)

# Resample the data to aggregate it by a specific time period (e.g., daily)
# Here, we'll resample the data on a daily basis and calculate the sum of 'transactionAmount' for each day
#daily_transactions = df['transactionAmount'].resample('D').sum()

daily_transactions = df.groupby('transactionDateTime')['transactionAmount'].sum()
# Plot the daily transaction amounts
plt.figure(figsize=(12, 6))
plt.bar(daily_transactions.index, daily_transactions.values)
plt.title('Daily Transaction Amounts Over Time')
plt.xlabel('Date')
plt.ylabel('Total Transaction Amount')
plt.show()




# *************** Correlation *********

corr=df[["creditLimit","availableMoney","transactionAmount","currentBalance","isFraud"]].corr()
sns.heatmap(corr,annot=True)


# log(0)=infinite ,so change 0 values to 0.001 so that we can get desire result



data1=df["creditLimit"]
data2=df["availableMoney"]
data3=df["transactionAmount"]
data4=df["currentBalance"]


#1 Log Transform
# data_log1=np.log(df.creditLimit)
# data_log2=np.log(df.availableMoney)
# data_log3=np.log(df.transactionAmount)
# data_log4=np.log(df.currentBalance)


fig,axs=plt.subplots(nrows=1,ncols=2)

# axs[0].hist(data1,edgecolor='black')
# axs[1].hist(data_log1,edgecolor='black')

# axs[0].hist(data2,edgecolor='black')
# axs[1].hist(data_log2,edgecolor='black')

# axs[0].hist(data3,edgecolor='black')        #infinite 
# axs[1].hist(data_log3,edgecolor='black')

# axs[0].hist(data4,edgecolor='black')    #infinite
# axs[1].hist(data_log4,edgecolor='black')



#2 Square root method

#works on percentage
data_log1=np.sqrt(df.creditLimit)
plt.hist(data_log1,edgecolor='black')

# datalog1=np.sqrt(data_log1)
# plt.hist(datalog1,edgecolor='black')

data_log2=np.sqrt(df.availableMoney)
plt.hist(data_log2,edgecolor='black')


data_log3=np.sqrt(df.transactionAmount)
plt.hist(data_log3,edgecolor='black')

data_log4=np.sqrt(df.currentBalance)
plt.hist(data_log4,edgecolor='black')





