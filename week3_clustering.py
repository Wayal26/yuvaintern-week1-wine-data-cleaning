import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

wine=load_wine(as_frame=True)
df=wine.frame.copy()
X=df.drop(columns=['target'])
X_scaled=StandardScaler().fit_transform(X)

results=[]
for k in range(2,11):
    model=KMeans(n_clusters=k,random_state=42,n_init=10)
    labels=model.fit_predict(X_scaled)
    results.append([k,model.inertia_,silhouette_score(X_scaled,labels)])
metrics=pd.DataFrame(results,columns=['k','inertia','silhouette'])
print(metrics)

model=KMeans(n_clusters=3,random_state=42,n_init=10)
df['cluster']=model.fit_predict(X_scaled)
print('Silhouette score:',silhouette_score(X_scaled,df['cluster']))

pca=PCA(n_components=2,random_state=42)
Xp=pca.fit_transform(X_scaled)
for c in sorted(df['cluster'].unique()):
    q=df['cluster'].values==c
    plt.scatter(Xp[q,0],Xp[q,1],label=f'Cluster {c}',alpha=.75)
plt.title('K-Means Clusters Visualized with PCA')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); plt.show()
