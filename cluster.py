import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import KMeans
#TITLE
st.set_page_config(page_title="KMeans",layout="centered")


#page styling

st.title("KMeans clustering Project")
st.write("Hello i am Arpita")
#data load

df=pd.read_csv("marks.csv")

#data preview

with st.expander("📈Data Preview"):
    st.dataframe(df)
x=df[["Study_hours","Attendance"]]

k=st.slider("enter your cluster",min_value=2,max_value=5)
model=KMeans(n_clusters=k,random_state=42,n_init=10)
df["cluster"]=model.fit_predict(x)
st.subheader("Cluster data")
st.dataframe(df)


st.subheader("cluster center")
centers=pd.DataFrame(model.cluster_centers_,columns=["Study Hours","Attendance"])
st.dataframe(df)

st.subheader("Student Cluster")
fig,ax=plt.subplots(figsize=(6,3))
scatter=ax.scatter(df["Study_hours"],df["Attendance"],c=df["cluster"],cmap="viridis")
ax.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],marker="*",color="red",s=100,label="centroids")

ax.set_title("KMeans Clustering")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Attendance")
ax.grid(True)
ax.legend()
st.pyplot(fig)