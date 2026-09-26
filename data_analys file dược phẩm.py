# Dự án thực hành Data Analysis Chuỗi cung ứng Dược phẩm
# Tham khảo logic từ tác giả: [https://www.kaggle.com/heardahe]
# Tùy chỉnh, sửa lỗi môi trường và tối ưu chạy Local bởi: [Nguyễn phong]

import pandas as pd
import numpy as np
# Đọc file dữ liệu
df = pd.read_csv('pharmacy_otc_sales_data.csv')

# Gom nhóm theo tên thuốc và tính tổng số hộp đã xuất kho
ton_kho_thuoc = df.groupby('Product')['Boxes Shipped'].sum().reset_index()
# Sắp xếp từ cao xuống thấp để tìm ra "vua doanh số số lượng"
ton_kho_thuoc = ton_kho_thuoc.sort_values(by='Boxes Shipped', ascending=False)

print("Báo cáo luân chuyển hàng hóa theo Sản phẩm:")
print(ton_kho_thuoc)
# Gom nhóm theo Quốc gia, tính tổng Số hộp và Doanh thu
logistics_vung = df.groupby('Country')[['Boxes Shipped', 'Amount ($)']].sum().reset_index()

# Sắp xếp theo tổng số hộp xuất đi
logistics_vung = logistics_vung.sort_values(by='Boxes Shipped', ascending=False)

print("\nBáo cáo tải trọng Logistics theo Khu vực:")
print(logistics_vung)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import plotly.express as px

df = pd.read_csv("pharmacy_otc_sales_data.csv")

print(df.info())
print(df.describe())
print(df.head(3))
print(df.tail(3))

print(f"The Number of Rows are {df.shape[0]}, and columns are {df.shape[1]}.")

df['Date'] = pd.to_datetime(df['Date'], format=None, errors='coerce')

df['Year']       = df['Date'].dt.year
df['Month']      = df['Date'].dt.month
df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)


# Quick null check
print("Null values:\n", df.isna().sum())

# Clean Amount column (remove currency symbols, convert to float)
df['Amount ($)'] = (
    df['Amount ($)']
      .replace(r'[\$,]', '', regex=True)
      .astype(float)
)

# 1. Compute unit cost (revenue per box) and add as a new column
df['Unit Cost ($/box)'] = df['Amount ($)'] / df['Boxes Shipped']

# Summary statistics for numeric fields
num_summary = df[['Boxes Shipped', 'Amount ($)']].describe().T
num_summary['total'] = df[['Boxes Shipped', 'Amount ($)']].sum()
print(num_summary)

#Unique values for Product, Sales Person, Country

for col in ['Product', 'Sales Person', 'Country']:
    uniques = df[col].unique()
    print(f"\n➡️ {col} ({len(uniques)} unique):")
    print(pd.Series(sorted(uniques), name=f"Unique {col}").to_frame())

    # Aggregate by product
prod_agg = (
    df.groupby('Product')
      .agg({'Boxes Shipped': 'sum', 'Amount ($)': 'sum'})
      .sort_values('Amount ($)', ascending=False)
      .reset_index()
)

# Plot top 5 by revenue
plt.figure(figsize=(10,5))
sns.barplot(
    data=prod_agg.head(5),
    x='Amount ($)', y='Product',
    palette='viridis'
)
plt.title("Top 5 Products by Revenue")
plt.xlabel("Total Revenue ($)")
plt.ylabel("")
plt.show()

# Plot top 5 by boxes shipped
plt.figure(figsize=(10,5))
sns.barplot(
    data=prod_agg.sort_values('Boxes Shipped', ascending=False).head(5),
    x='Boxes Shipped', y='Product',
    palette='magma'
)
plt.title("Top 5 Products by Volume")
plt.xlabel("Total Boxes Shipped")
plt.ylabel("")
plt.show()

# Aggregate by sales person
sp_agg = (
    df.groupby('Sales Person')
      .agg({'Boxes Shipped': 'sum', 'Amount ($)': 'sum'})
      .sort_values('Amount ($)', ascending=False)
      .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14,5))
sns.barplot(
    data=sp_agg, x='Amount ($)', y='Sales Person',
    ax=axes[0], palette='rocket'
)
axes[0].set_title("Revenue by Sales Person")
axes[0].set_xlabel("Total Revenue ($)")
axes[0].set_ylabel("")

sns.barplot(
    data=sp_agg, x='Boxes Shipped', y='Sales Person',
    ax=axes[1], palette='flare'
)
axes[1].set_title("Volume by Sales Person")
axes[1].set_xlabel("Total Boxes Shipped")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

# Monthly aggregation
monthly = (
    df.groupby('Month_Year')
      .agg({'Boxes Shipped': 'sum', 'Amount ($)': 'sum'})
      .reset_index()
)

# Convert Month_Year back to datetime for plotting
monthly['Month_Year'] = pd.to_datetime(monthly['Month_Year'])

# Plot trend
plt.figure(figsize=(12,6))
sns.lineplot(
    data=monthly, x='Month_Year', y='Amount ($)',
    marker='o', label='Revenue'
)

plt.title("Monthly Revenue")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# Country aggregation
cty_agg = (
    df.groupby('Country')
      .agg({'Boxes Shipped': 'sum', 'Amount ($)': 'sum'})
      .sort_values('Amount ($)', ascending=False)
      .reset_index()
)

plt.figure(figsize=(10,6))
sns.barplot(
    data=cty_agg, x='Amount ($)', y='Country',
    palette='crest'
)
plt.title("Revenue by Country")
plt.xlabel("Total Revenue ($)")
plt.ylabel("")
plt.show()

# Aggregate across company, compute average unit price by product
prod_unit = (
    df.groupby('Product')
      .agg(Total_Boxes=('Boxes Shipped','sum'),
           Total_Revenue=('Amount ($)','sum'))
      .assign(
          Avg_Unit_Price=lambda d: d['Total_Revenue']/d['Total_Boxes']
      )
      .sort_values('Avg_Unit_Price', ascending=False)
      .reset_index()
)

# Select top 5 by unit price
top5_unit = prod_unit.head(5)

# Show table
print(top5_unit[['Product','Total_Boxes','Total_Revenue','Avg_Unit_Price']])

# Bar chart of top 5
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.barplot(
    data=top5_unit,
    x='Avg_Unit_Price', y='Product',
    palette='viridis'
)
plt.title("Top 5 Products by Company Unit Price ($/box)")
plt.xlabel("Avg Unit Price ($)")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Filter original data to just those top 5 products
top_products = top5_unit['Product'].tolist()
df_top = df[df['Product'].isin(top_products)].copy()

# Aggregate performance by Sales Person & Product
perf = (
    df_top
      .groupby(['Sales Person','Product'])
      .agg(Revenue=('Unit Cost ($/box)','mean'))
      .reset_index())

# Pivot for a heatmap or grouped bar chart
perf_pivot = perf.pivot(
    index='Sales Person',
    columns='Product',
    values='Revenue'
).fillna(0)

# Display pivot
print(perf_pivot)

# Plot grouped bars for revenue
perf_melt = perf  # already in long form
plt.figure(figsize=(10,6))
sns.barplot(
    data=perf_melt,
    x='Revenue', y='Sales Person',
    hue='Product'
)
plt.title("Sales Person Revenue on Top 5 High-Price Products")
plt.xlabel("Unit Price")
plt.ylabel("")
plt.legend(title="Product", bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()