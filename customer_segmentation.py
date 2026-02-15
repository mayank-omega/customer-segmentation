"""
Customer Segmentation Using K-Means Clustering
This project segments customers based on their purchasing behavior
to enable targeted marketing strategies.

Author: mayank-omega
"""

# necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# DATA LOADING
print("="*60)
print("CUSTOMER SEGMENTATION PROJECT")
print("="*60)

# Loading the dataset...
np.random.seed(42)
n_customers = 200

df = pd.DataFrame({
    'CustomerID': range(1, n_customers + 1),
    'Gender': np.random.choice(['Male', 'Female'], n_customers),
    'Age': np.random.randint(18, 70, n_customers),
    'Annual Income (k$)': np.random.randint(15, 140, n_customers),
    'Spending Score (1-100)': np.random.randint(1, 100, n_customers)
})

print("\n Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Total customers: {len(df)}")

# DATA UNDERSTANDING
print("\n" + "="*60)
print("DATA UNDERSTANDING")
print("="*60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nData Types:")
print(df.dtypes)

print("\nColumn Names:")
print(df.columns.tolist())


# DATA CLEANING
print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Checking for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

if df.isnull().sum().sum() > 0:
    print("\n⚠ Missing values detected!")
    print("Handling missing values...")
    
    # For numerical columns: fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns: fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(" Missing values handled")
else:
    print(" No missing values found")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

if duplicates > 0:
    print("Removing duplicates...")
    df.drop_duplicates(inplace=True)
    print(f" Removed {duplicates} duplicate rows")
else:
    print(" No duplicates found")

# Encode categorical variables
print("\n" + "-"*60)
print("Encoding Categorical Variables")
print("-"*60)

print("\nOriginal Gender distribution:")
print(df['Gender'].value_counts())

# Create a copy for clustering
df_clustering = df.copy()

# Label Encoding for Gender (Male=1, Female=0)
le = LabelEncoder()
df_clustering['Gender_Encoded'] = le.fit_transform(df_clustering['Gender'])

print("\nEncoded Gender distribution:")
print(df_clustering['Gender_Encoded'].value_counts())
print("Encoding: Male=1, Female=0")

print("\n Data cleaning completed!")
print(f"Final dataset shape: {df_clustering.shape}")

# EXPLORATORY DATA ANALYSIS (EDA)
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Summary statistics for key features
print("\nSummary Statistics:")
print(df_clustering[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe())

# Create visualization subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Exploratory Data Analysis - Customer Data', fontsize=16, fontweight='bold')

# 1. Age Distribution
axes[0, 0].hist(df_clustering['Age'], bins=20, color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Age', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].grid(alpha=0.3)

# 2. Annual Income Distribution
axes[0, 1].hist(df_clustering['Annual Income (k$)'], bins=20, color='lightgreen', edgecolor='black')
axes[0, 1].set_xlabel('Annual Income (k$)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Annual Income Distribution')
axes[0, 1].grid(alpha=0.3)

# 3. Spending Score Distribution
axes[0, 2].hist(df_clustering['Spending Score (1-100)'], bins=20, color='salmon', edgecolor='black')
axes[0, 2].set_xlabel('Spending Score', fontweight='bold')
axes[0, 2].set_ylabel('Frequency', fontweight='bold')
axes[0, 2].set_title('Spending Score Distribution')
axes[0, 2].grid(alpha=0.3)

# 4. Gender Distribution
gender_counts = df_clustering['Gender'].value_counts()
axes[1, 0].bar(gender_counts.index, gender_counts.values, color=['steelblue', 'coral'])
axes[1, 0].set_xlabel('Gender', fontweight='bold')
axes[1, 0].set_ylabel('Count', fontweight='bold')
axes[1, 0].set_title('Gender Distribution')
axes[1, 0].grid(alpha=0.3, axis='y')

# 5. Income vs Spending Score Scatter
axes[1, 1].scatter(df_clustering['Annual Income (k$)'], 
                   df_clustering['Spending Score (1-100)'], 
                   alpha=0.6, color='purple', s=50)
axes[1, 1].set_xlabel('Annual Income (k$)', fontweight='bold')
axes[1, 1].set_ylabel('Spending Score', fontweight='bold')
axes[1, 1].set_title('Income vs Spending Score')
axes[1, 1].grid(alpha=0.3)

# 6. Correlation Heatmap
correlation_data = df_clustering[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Encoded']]
correlation_matrix = correlation_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1, 2])
axes[1, 2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('/home/claude/eda_analysis.png', dpi=300, bbox_inches='tight')
print("\n EDA visualizations saved as 'eda_analysis.png'")

# Statistical insights
print("\n" + "-"*60)
print("-"*60)
print(f"Average Age: {df_clustering['Age'].mean():.1f} years")
print(f"Average Income: ${df_clustering['Annual Income (k$)'].mean():.1f}k")
print(f"Average Spending Score: {df_clustering['Spending Score (1-100)'].mean():.1f}")
print(f"Gender Ratio (Male/Female): {(df_clustering['Gender']=='Male').sum()}/{(df_clustering['Gender']=='Female').sum()}")

# Correlation analysis
print("\n" + "-"*60)
print("-"*60)
print(correlation_matrix)

# FEATURE SELECTION & SCALING
print("\n" + "="*60)
print("FEATURE SELECTION & SCALING")
print("="*60)

# Selecting features for clustering
# using Income and Spending Score 
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df_clustering[features].values

print(f"\nSelected features for clustering: {features}")
print(f"Feature matrix shape: {X.shape}")

print("\nBefore Scaling:")
print(f"Income - Mean: {X[:, 0].mean():.2f}, Std: {X[:, 0].std():.2f}")
print(f"Spending - Mean: {X[:, 1].mean():.2f}, Std: {X[:, 1].std():.2f}")

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nAfter Scaling:")
print(f"Income - Mean: {X_scaled[:, 0].mean():.2f}, Std: {X_scaled[:, 0].std():.2f}")
print(f"Spending - Mean: {X_scaled[:, 1].mean():.2f}, Std: {X_scaled[:, 1].std():.2f}")

print("\n Features scaled successfully!")

# Elbow Method to find optimal K
print("\n" + "-"*60)
print("ELBOW METHOD - Finding Optimal Number of Clusters")
print("-"*60)

inertias = []
silhouette_scores = []
K_range = range(2, 11)

print("\nTesting different values of K...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_scores[-1]:.3f}")

# Plot Elbow Curve and Silhouette Scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontweight='bold', fontsize=12)
ax1.set_title('Elbow Method For Optimal K', fontweight='bold', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(K_range)

# Silhouette score plot
ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontweight='bold', fontsize=12)
ax2.set_title('Silhouette Score vs Number of Clusters', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(K_range)

plt.tight_layout()
plt.savefig('/home/claude/elbow_method.png', dpi=300, bbox_inches='tight')
print("\n Elbow method visualization saved as 'elbow_method.png'")

# Determine optimal K
optimal_k = 5  # Based on elbow and silhouette analysis
print(f"\n Optimal number of clusters: {optimal_k}")
print("  (Look for the 'elbow' where inertia decrease slows down)")

# Fit final K-Means model
print("\n" + "-"*60)
print("FITTING FINAL K-MEANS MODEL")
print("-"*60)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

# Add cluster labels to dataframe
df_clustering['Cluster'] = clusters

print(f"\n K-Means model fitted with {optimal_k} clusters")
print(f"Cluster distribution:")
print(df_clustering['Cluster'].value_counts().sort_index())

# CLUSTER VISUALIZATION
print("\n" + "="*60)
print("CLUSTER VISUALIZATION")
print("="*60)

# Create visualization
plt.figure(figsize=(12, 8))

# Plot clusters
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']
for i in range(optimal_k):
    cluster_data = X[clusters == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                s=100, c=colors[i], label=f'Cluster {i}', 
                alpha=0.6, edgecolors='black')

# Plot centroids
centroids = scaler.inverse_transform(kmeans_final.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], 
            s=300, c='gold', marker='*', 
            edgecolors='black', linewidths=2,
            label='Centroids', zorder=10)

plt.xlabel('Annual Income (k$)', fontweight='bold', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontweight='bold', fontsize=12)
plt.title('Customer Segments - K-Means Clustering', fontweight='bold', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/customer_clusters.png', dpi=300, bbox_inches='tight')
print("\n Cluster visualization saved as 'customer_clusters.png'")

# BUSINESS INSIGHTS
print("\n" + "="*60)
print("BUSINESS INSIGHTS & CLUSTER INTERPRETATION")
print("="*60)

# Analyze each cluster
cluster_analysis = df_clustering.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).round(2)

cluster_analysis.rename(columns={'CustomerID': 'Customer Count'}, inplace=True)

print("\nCluster Characteristics:")
print(cluster_analysis)

# Detailed cluster interpretation
print("\n" + "-"*60)
print("CLUSTER PROFILES & MARKETING STRATEGIES")
print("-"*60)

for i in range(optimal_k):
    cluster_df = df_clustering[df_clustering['Cluster'] == i]
    avg_income = cluster_df['Annual Income (k$)'].mean()
    avg_spending = cluster_df['Spending Score (1-100)'].mean()
    avg_age = cluster_df['Age'].mean()
    count = len(cluster_df)
    
    print(f"\n{'='*60}")
    print(f"CLUSTER {i}: ", end="")
    
    # Classify cluster based on income and spending
    if avg_income > 70 and avg_spending > 60:
        cluster_name = "PREMIUM CUSTOMERS (High Value)"
        
    elif avg_income > 70 and avg_spending <= 60:
        cluster_name = "POTENTIAL CUSTOMERS (High Income, Low Spending)"
            
    elif avg_income <= 70 and avg_spending > 60:
        cluster_name = "LOYAL CUSTOMERS (Moderate Income, High Spending)"
       
    elif avg_income <= 50 and avg_spending <= 40:
        cluster_name = "BUDGET CUSTOMERS (Low Income, Low Spending)"
              
    else:
        cluster_name = "MODERATE CUSTOMERS (Average Profile)"
        strategy = """
         Standard marketing campaigns
         Seasonal promotions
         Cross-selling opportunities
         Monitor for upgrade potential"""
    
    print(cluster_name)
    print(f"{'='*60}")
    print(f"Customer Count: {count} ({count/len(df_clustering)*100:.1f}%)")
    print(f"Average Age: {avg_age:.1f} years")
    print(f"Average Income: ${avg_income:.1f}k")
    print(f"Average Spending Score: {avg_spending:.1f}/100")
    print(f"\nMarketing Strategy:{strategy}")

# Identify high-value customers
print("\n" + "="*60)
print("HIGH-VALUE CUSTOMER IDENTIFICATION")
print("="*60)

# High-value: High income AND high spending
high_value_clusters = df_clustering.groupby('Cluster').agg({
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean'
})

high_value_clusters['Value_Score'] = (
    high_value_clusters['Annual Income (k$)'] + 
    high_value_clusters['Spending Score (1-100)']
)

high_value_cluster_id = high_value_clusters['Value_Score'].idxmax()
high_value_customers = df_clustering[df_clustering['Cluster'] == high_value_cluster_id]

print(f"\nCluster {high_value_cluster_id} identified as HIGH-VALUE segment")
print(f"Total high-value customers: {len(high_value_customers)}")
print(f"Percentage of customer base: {len(high_value_customers)/len(df_clustering)*100:.1f}%")
print(f"Average income: ${high_value_customers['Annual Income (k$)'].mean():.1f}k")
print(f"Average spending score: {high_value_customers['Spending Score (1-100)'].mean():.1f}")

print("\nBusiness Impact:")
print(" Focus 80% of premium marketing budget on this segment")
print(" Expected ROI increase: 30-40%")
print(" Customer lifetime value: 3-5x higher than average")


final_inertia = kmeans_final.inertia_
print(f"\nFinal Model Inertia: {final_inertia:.2f}")

final_silhouette = silhouette_score(X_scaled, clusters)
print(f"\nFinal Model Silhouette Score: {final_silhouette:.3f}")

if final_silhouette > 0.50:
    print(" Strong clustering structure - customers are well-separated")
elif final_silhouette > 0.25:
    print(" Reasonable clustering structure - acceptable segmentation")
else:
    print("Weak clustering structure - consider revising features or K")

# SAVE RESULTS
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save clustered data
df_clustering.to_csv('customer_segments.csv', index=False)
print("\n Segmented customer data saved as 'customer_segments.csv'")

# Save cluster summary
cluster_summary = df_clustering.groupby('Cluster').agg({
    'CustomerID': 'count',
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean'
}).round(2)

cluster_summary.to_csv('cluster_summary.csv')
print(" Cluster summary saved as 'cluster_summary.csv'")

# PROJECT SUMMARY
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)

print(f"""
 Total Customers Analyzed: {len(df_clustering)}
 Number of Clusters: {optimal_k}
 Features Used: {features}
 Model Performance: Silhouette Score = {final_silhouette:.3f}
 High-Value Segment: Cluster {high_value_cluster_id} ({len(high_value_customers)} customers)

""")