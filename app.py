import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from traitlets import default

st.set_page_config(
    page_title="House Clustering App",
    page_icon="🏠",
    layout="wide"
)

st.title("House Clustering Using K-Means App")
st.write("Machine Learning Project - Clustering Houses Based on Features")

uploaded_file=st.file_uploader(
    "Upload your CSV file",
     type=["csv"])

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview:")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(
        include=[np.number]).columns.tolist()

    default_features=[
        col for col in[
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "longitude",
            "latitude"
        ] if col in numeric_columns
        
    ]

    st.subheader("Select Features for Clustering:")

    features = st.multiselect(
        "Select Features for Clustering",
        options=numeric_columns,
        default=numeric_columns[:4]
    )

    if len(features) >= 2:

        data=df[features].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 3:
            st.error("Not enough data after removing NaN and infinite values. Please check your dataset.")
            st.stop()

        scaler=StandardScaler()
        data_scaled=scaler.fit_transform(data)

        st.subheader("Select Number of Clusters:")

        k=st.slider(
            "Number of clusters (K)",
            min_value=2,
            max_value=8,
            value=5
        )

        model=KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        clusters=model.fit_predict(data_scaled)

        result=df.loc[data.index].copy()
        result['Cluster'] = clusters

        st.subheader("Clustering Result:")
        st.dataframe(result.head(20))

        unique_clusters = np.unique(clusters)

        if 1<len(unique_clusters)<len(data):
            score=silhouette_score(
            data_scaled, clusters
        )

        st.metric(
            "Silhouette Score",
            round(score, 3)
        )

        st.subheader("Cluster Distribution:")

        counts=(
            pd.Series(clusters)
            .value_counts()
            .reindex(range(k), fill_value=0)
        )

        st.bar_chart(counts)
            

        st.subheader("House Clusters Visualization (PCA):")

        pca=PCA(n_components=2)
        reduced_data=pca.fit_transform(data_scaled)

        fig, ax=plt.subplots()

        scatter=ax.scatter(
            
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=clusters,
            cmap='viridis',
            s=12,
            alpha=0.6
        )

        ax.set_title("K-Means House Clusters")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")

        fig.colorbar(scatter, label='Cluster')
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        st.header("House Price Category Analysis")
        price_column="median_house_value"

        if price_column in df.columns:
            price_summary=(
                result.groupby("Cluster")[price_column]
                .mean()
                .reindex(range(k))
            )
            valid_prices= price_summary.dropna()

            if len(valid_prices) > 0:
                low_limit=valid_prices.quantile(0.33)
                high_limit=valid_prices.quantile(0.66)

                def get_category(price):
                    if price < low_limit:
                        return "Low"
                    elif price < high_limit:
                        return "Medium"
                    else:
                        return "High"

                categories={
                        cluster: get_category(price)
                        for cluster, price in price_summary.items()
                    }
                price_table=pd.DataFrame({
                    "Cluster":valid_prices.index,
                    "Average Price":valid_prices.values,
                    "Price Category":[categories[c] for c in valid_prices.index]
                })
                price_table["Average Price"] = (
                    price_table["Average Price"].map(
                        lambda x: f"${x:,.0f}"
                    )
                )
                st.write(
                    "Houses are classified using the average "
                    "house value of each cluster."
                )
                st.dataframe(
                    price_table,
                    use_container_width=True,
                    hide_index=True
                )


                st.subheader("Download Clustered Dataset:")

                category_counts=(
                    pd.Series(categories)
                    .value_counts()
                    .reindex(["Low", "Medium", "High"], fill_value=0)
                )
                st.bar_chart(category_counts)

                st.info(
                    "These categories are relative to the uploaded "
                    "dataset, not fixed real-estate market price ranges."
                )

            else:
                st.warning("No valid house prices found.")

        else:
            categories = None

            st.warning(
                "The dataset does not contain median_house_value. "
                "Price categories cannot be calculated."
            )
        st.divider()

        st.header("Find the Cluster of a New House")

        st.write(
            "Enter the details of a new house. "
            "The trained K-Means model will identify its cluster."
        )

        with st.form("house_prediction_form"):

            st.subheader("Enter House Details")

            new_house = {}

            columns = st.columns(2)

            for i, feature in enumerate(features):

                minimum = float(data[feature].min())
                maximum = float(data[feature].max())
                median = float(data[feature].median())

                if minimum == maximum:
                    minimum -= 1
                    maximum += 1

                with columns[i % 2]:

                    new_house[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(median, minimum), maximum),
                        key=f"new_{feature}"
                    )

            submitted = st.form_submit_button(
                "Identify House Cluster"
            )

        if submitted:

            new_data = pd.DataFrame(
                [new_house],
                columns=features
            )

            new_scaled = scaler.transform(new_data)

            predicted_cluster = int(
                model.predict(new_scaled)[0]
            )

            st.success(
                f"New house belongs to Cluster {predicted_cluster}"
            )

            if (
                "categories" in locals()
                and categories is not None
                and predicted_cluster in categories
            ):

                category = categories[predicted_cluster]

                if category == "Low Value":
                    st.info("🏠 Price Category: Low Value")

                elif category == "Medium Value":
                    st.warning("🏠 Price Category: Medium Value")

                else:
                    st.success("🏠 Price Category: High Value")

                avg_price = valid_prices.loc[predicted_cluster]

                st.metric(
                    "Average Value of This Cluster",
                    f"${avg_price:,.0f}"
                )

            else:
                st.info(
                    "Price category is unavailable for this dataset."
                )

        st.divider()

        st.subheader("Download Clustered Dataset:")

        csv=result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            data=csv,
            file_name="house_clusters.csv",
            mime="text/csv",
        )

    else:
        st.warning("Please select at least 2 numeric features for clustering.")

else:
    st.info("Please upload a CSV Dataset to proceed with clustering.")



