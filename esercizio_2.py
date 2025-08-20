#Mi leggi un file Online_Retail.csv

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

df = pd.read_csv('Online_Retail.csv', encoding='latin1')
print(df.head())

# Calcola il totale speso per ogni riga
df['TotalSpent'] = df['Quantity'] * df['UnitPrice']

# Raggruppa per CustomerID e somma il totale speso
df_customer = df.groupby('CustomerID')['TotalSpent'].sum().reset_index()

# Rinomina la colonna
df_customer.rename(columns={'TotalSpent': 'CustomerLifetimeValue'}, inplace=True)


#print(df_customer.head(20))

print("\n")

# Trova il numero totale di acquisti per ogni cliente
total_purchases = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
total_purchases.rename(columns={'InvoiceNo': 'TotalPurchases'}, inplace=True)

# Aggiungi la colonna TotalPurchases al dataframe originale
df_customer = df_customer.merge(total_purchases, on='CustomerID', how='left')

# Trova la spesa media per ogni acquisto per ogni cliente
average_spent = df.groupby('CustomerID')['TotalSpent'].mean().reset_index(name='AverageSpent')
df_customer = df_customer.merge(average_spent, on='CustomerID', how='left')

# Calcola la frequenza di acquisto su base mensile

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Estrai anno e mese
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')

# Conta il numero di acquisti (fatture uniche) per cliente per ogni mese
monthly_purchases = df.groupby(['CustomerID', 'YearMonth'])['InvoiceNo'].nunique().reset_index(name='MonthlyPurchases')

# Calcola la frequenza media mensile per ciascun cliente
avg_monthly_frequency = monthly_purchases.groupby('CustomerID')['MonthlyPurchases'].mean().reset_index(name='AverageMonthlyPurchases')
df_customer = df_customer.merge(avg_monthly_frequency, on='CustomerID', how='left')

# Per ogni cliente riportami il paese di appartenenza
country = df.groupby('CustomerID')['Country'].first().reset_index()
df_customer = df_customer.merge(country, on='CustomerID', how='left')

#print(df_customer.head(20))

# Mi normalizzi i dati in media nulla e varianza unitaria

scaler = StandardScaler()
df_customer[['CustomerLifetimeValue', 'TotalPurchases', 'AverageSpent', 'AverageMonthlyPurchases']] = scaler.fit_transform(df_customer[['CustomerLifetimeValue', 'TotalPurchases', 'AverageSpent', 'AverageMonthlyPurchases']])

print(df_customer.head(10))

# Applica K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df_customer['Cluster'] = kmeans.fit_predict(df_customer)

# Mi plotti i punti ed i cluster con matplotlib

plt.figure(figsize=(10, 6))
plt.scatter(df_customer['CustomerLifetimeValue'], df_customer['TotalPurchases'], c=df_customer['Cluster'], cmap='viridis')
plt.xlabel('Customer Lifetime Value')
plt.ylabel('Total Purchases')
plt.title('Customer Segmentation')
plt.colorbar(label='Cluster')
plt.show() 

# Applica DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
df_customer['DBSCAN_Cluster'] = dbscan.fit_predict(df_customer)

# Mi plotti i punti ed i cluster con matplotlib

plt.figure(figsize=(10, 6))
plt.scatter(df_customer['CustomerLifetimeValue'], df_customer['TotalPurchases'], c=df_customer['DBSCAN_Cluster'], cmap='viridis')
plt.xlabel('Customer Lifetime Value')
plt.ylabel('Total Purchases')
plt.title('Customer Segmentation - DBSCAN')
plt.colorbar(label='Cluster')
plt.show()

