import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Mobile Product Segmentation",
    page_icon="📱",
    layout="wide"
)


# =========================================================
# 2. LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    data = pd.read_csv("mobile_products_clustered.csv")

    # Remove accidental spaces from column names
    data.columns = data.columns.str.strip()

    return data


try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "Dataset not found. Place mobile_products_clustered.csv "
        "in the same folder as app.py."
    )
    st.stop()


# =========================================================
# 3. DATA VALIDATION
# =========================================================

required_columns = [
    "brand",
    "model",
    "price_usd",
    "rating"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()


# =========================================================
# 4. COMMON VARIABLES
# =========================================================

numeric_features = [
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]

numeric_features = [
    col for col in numeric_features
    if col in df.columns
]

# Convert relevant columns to numeric
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📱 Mobile Analytics")

st.sidebar.write(
    "Explore mobile products, customer ratings, "
    "product segments, and recommendations."
)

page = st.sidebar.radio(
    "Navigate to",
    [
        "Home",
        "EDA",
        "Product Segmentation",
        "Recommendations",
        "Business Insights"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Mobile Product Segmentation and Recommendation System"
)


# =========================================================
# 6. HOME PAGE
# =========================================================

if page == "Home":

    st.title(
        "📱 Mobile Product Segmentation "
        "and Recommendation System"
    )

    st.write(
        """
        This dashboard analyzes mobile product reviews,
        explores customer ratings, identifies product segments,
        and recommends similar mobile models.
        """
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Review Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Mobile Brands",
            df["brand"].nunique()
        )

    with col3:
        st.metric(
            "Mobile Models",
            df["model"].nunique()
        )

    with col4:
        if "cluster" in df.columns:
            st.metric(
                "Clusters",
                df["cluster"].nunique()
            )
        else:
            st.metric("Clusters", "Not available")

    st.divider()

    # Overview
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Average Mobile Price")

        avg_price = df["price_usd"].mean()

        st.metric(
            "Average Price",
            f"${avg_price:,.2f}"
            if pd.notna(avg_price)
            else "N/A"
        )

    with col2:
        st.subheader("⭐ Average Customer Rating")

        avg_rating = df["rating"].mean()

        st.metric(
            "Average Rating",
            f"{avg_rating:.2f} / 5"
            if pd.notna(avg_rating)
            else "N/A"
        )

    st.divider()

    # Brand distribution
    st.subheader("📊 Mobile Reviews by Brand")

    brand_counts = (
        df["brand"]
        .value_counts()
        .head(10)
        .rename_axis("brand")
        .reset_index(name="review_count")
    )

    fig = px.bar(
        brand_counts,
        x="brand",
        y="review_count",
        title="Top 10 Brands by Review Count",
        labels={
            "brand": "Brand",
            "review_count": "Number of Reviews"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# 7. EXPLORATORY DATA ANALYSIS
# =========================================================

elif page == "EDA":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "Explore brand distribution, customer ratings, "
        "mobile prices, and relationships between features."
    )

    # -----------------------------------------------------
    # Chart 1: Brand distribution
    # -----------------------------------------------------

    st.subheader("1. Review Distribution by Brand")

    brand_counts = (
        df["brand"]
        .value_counts()
        .head(10)
        .rename_axis("brand")
        .reset_index(name="review_count")
    )

    fig1 = px.bar(
        brand_counts,
        x="brand",
        y="review_count",
        title="Top 10 Brands by Review Count",
        labels={
            "brand": "Brand",
            "review_count": "Review Count"
        }
    )

    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------------------------------
    # Chart 2: Average rating by brand
    # -----------------------------------------------------

    st.subheader("2. Average Customer Rating by Brand")

    rating_by_brand = (
        df.groupby("brand", as_index=False)["rating"]
        .mean()
        .sort_values("rating", ascending=False)
    )

    fig2 = px.bar(
        rating_by_brand,
        x="brand",
        y="rating",
        title="Average Rating by Brand",
        labels={
            "brand": "Brand",
            "rating": "Average Rating"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------
    # Chart 3: Price distribution
    # -----------------------------------------------------

    st.subheader("3. Mobile Price Distribution")

    fig3 = px.histogram(
        df,
        x="price_usd",
        nbins=30,
        title="Distribution of Mobile Prices",
        labels={
            "price_usd": "Price (USD)"
        }
    )

    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------------------------------
    # Chart 4: Rating distribution
    # -----------------------------------------------------

    st.subheader("4. Customer Rating Distribution")

    fig4 = px.histogram(
        df,
        x="rating",
        nbins=20,
        title="Distribution of Customer Ratings",
        labels={
            "rating": "Customer Rating"
        }
    )

    st.plotly_chart(fig4, use_container_width=True)

    # -----------------------------------------------------
    # Chart 5: Price vs rating
    # -----------------------------------------------------

    st.subheader("5. Relationship Between Price and Rating")

    color_column = (
        "cluster"
        if "cluster" in df.columns
        else "brand"
    )

    fig5 = px.scatter(
        df,
        x="price_usd",
        y="rating",
        color=color_column,
        hover_data=["brand", "model"],
        title="Mobile Price vs Customer Rating",
        labels={
            "price_usd": "Price (USD)",
            "rating": "Customer Rating",
            color_column: color_column.title()
        }
    )

    st.plotly_chart(fig5, use_container_width=True)

    # -----------------------------------------------------
    # Statistical summary
    # -----------------------------------------------------

    st.subheader("6. Statistical Summary")

    st.dataframe(
        df[numeric_features].describe().round(2),
        use_container_width=True
    )

    # -----------------------------------------------------
    # Correlation matrix
    # -----------------------------------------------------

    if len(numeric_features) >= 2:

        st.subheader("7. Feature Correlation")

        correlation = df[numeric_features].corr()

        fig6 = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Between Numerical Features",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1
        )

        st.plotly_chart(fig6, use_container_width=True)


# =========================================================
# 8. PRODUCT SEGMENTATION
# =========================================================

elif page == "Product Segmentation":

    st.title("🎯 Product Segmentation")

    st.write(
        """
        This section explores the clusters created using
        the K-Means clustering algorithm.
        """
    )

    if "cluster" not in df.columns:

        st.warning(
            "The dataset does not contain a cluster column. "
            "Run your clustering code and save the clustered dataset first."
        )

    else:

        # Cluster distribution
        st.subheader("1. Number of Records in Each Cluster")

        cluster_counts = (
            df["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("cluster")
            .reset_index(name="record_count")
        )

        fig1 = px.bar(
            cluster_counts,
            x="cluster",
            y="record_count",
            title="Cluster Distribution",
            labels={
                "cluster": "Cluster",
                "record_count": "Number of Records"
            }
        )

        st.plotly_chart(fig1, use_container_width=True)

        # Price vs rating by cluster
        st.subheader("2. Compare Clusters by Price and Rating")

        fig2 = px.scatter(
            df,
            x="price_usd",
            y="rating",
            color="cluster",
            hover_data=["brand", "model"],
            title="Price and Rating Across Clusters",
            labels={
                "price_usd": "Price (USD)",
                "rating": "Customer Rating",
                "cluster": "Cluster"
            }
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Cluster profiles
        st.subheader("3. Cluster Characteristics")

        profile_features = [
            col for col in numeric_features
            if col in df.columns
        ]

        cluster_profile = (
            df.groupby("cluster")[profile_features]
            .mean()
            .round(2)
        )

        st.dataframe(
            cluster_profile,
            use_container_width=True
        )

        # Average price by cluster
        st.subheader("4. Average Price by Cluster")

        price_by_cluster = (
            df.groupby("cluster", as_index=False)["price_usd"]
            .mean()
        )

        fig3 = px.bar(
            price_by_cluster,
            x="cluster",
            y="price_usd",
            title="Average Price of Each Cluster",
            labels={
                "cluster": "Cluster",
                "price_usd": "Average Price (USD)"
            }
        )

        st.plotly_chart(fig3, use_container_width=True)

        # Average rating by cluster
        st.subheader("5. Average Rating by Cluster")

        rating_by_cluster = (
            df.groupby("cluster", as_index=False)["rating"]
            .mean()
        )

        fig4 = px.bar(
            rating_by_cluster,
            x="cluster",
            y="rating",
            title="Average Rating of Each Cluster",
            labels={
                "cluster": "Cluster",
                "rating": "Average Rating"
            }
        )

        st.plotly_chart(fig4, use_container_width=True)

        st.info(
            """
            Important: A cluster number such as 0, 1, or 2 is just
            a label assigned by the algorithm. Interpret each cluster
            using its average price, ratings, and specifications.

            Do not automatically label a cluster as premium or budget
            without examining its characteristics.
            """
        )


# =========================================================
# 9. RECOMMENDATION SYSTEM
# =========================================================

elif page == "Recommendations":

    st.title("📱 Mobile Product Recommendations")

    st.write(
        """
        Select a mobile model to discover other models with
        similar numerical characteristics using cosine similarity.
        """
    )

    # Aggregate review records to model-level data
    model_columns = ["model"]

    if "brand" in df.columns:
        model_columns.append("brand")

    if "country" in df.columns:
        model_columns.append("country")

    available_features = [
        col for col in numeric_features
        if col in df.columns
    ]

    # Mean feature values for each model
    model_data = (
        df.groupby(model_columns, as_index=False)[available_features]
        .mean()
    )

    if len(model_data) < 2:

        st.warning(
            "At least two distinct mobile models are required "
            "to generate recommendations."
        )

    else:

        # Features used to compare models
        recommendation_features = [
            col for col in available_features
            if model_data[col].notna().any()
        ]

        if not recommendation_features:

            st.warning(
                "No suitable numerical features are available."
            )

        else:

            # Fill missing values using each feature's median
            X = model_data[recommendation_features].copy()

            X = X.fillna(X.median())

            # Remove features that have no variation
            recommendation_features = [
                col for col in X.columns
                if X[col].nunique() > 1
            ]

            X = X[recommendation_features]

            if X.shape[1] == 0:

                st.warning(
                    "The available features do not vary enough "
                    "to calculate meaningful recommendations."
                )

            else:

                # Scale features before similarity calculation
                scaler = StandardScaler()

                X_scaled = scaler.fit_transform(X)

                similarity_matrix = cosine_similarity(X_scaled)

                selected_model = st.selectbox(
                    "Choose a mobile model",
                    sorted(model_data["model"].dropna().unique())
                )

                top_n = st.slider(
                    "Number of recommendations",
                    min_value=1,
                    max_value=min(10, len(model_data) - 1),
                    value=min(5, len(model_data) - 1)
                )

                if st.button("Find Similar Models"):

                    selected_indices = model_data.index[
                        model_data["model"] == selected_model
                    ].tolist()

                    if not selected_indices:

                        st.warning("Selected model was not found.")

                    else:

                        selected_index = selected_indices[0]

                        similarity_scores = list(
                            enumerate(
                                similarity_matrix[selected_index]
                            )
                        )

                        # Exclude the selected model itself
                        similarity_scores = [
                            (idx, score)
                            for idx, score in similarity_scores
                            if idx != selected_index
                        ]

                        similarity_scores.sort(
                            key=lambda item: item[1],
                            reverse=True
                        )

                        recommended_indices = [
                            idx
                            for idx, score in similarity_scores[:top_n]
                        ]

                        recommended_scores = [
                            score
                            for idx, score in similarity_scores[:top_n]
                        ]

                        recommendations = model_data.iloc[
                            recommended_indices
                        ].copy()

                        recommendations["similarity_score"] = (
                            recommended_scores
                        )

                        recommendations["similarity_score"] = (
                            recommendations["similarity_score"]
                            .round(3)
                        )

                        st.subheader(
                            f"Recommendations for {selected_model}"
                        )

                        st.dataframe(
                            recommendations,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.success(
                            "Recommendations generated successfully."
                        )

                        st.caption(
                            """
                            Similarity is calculated from standardized
                            numerical features. A higher cosine similarity
                            indicates greater similarity in the selected
                            feature space. These recommendations should
                            be validated for practical relevance.
                            """
                        )


# =========================================================
# 10. BUSINESS INSIGHTS
# =========================================================

elif page == "Business Insights":

    st.title("💡 Business Insights and Findings")

    st.write(
        """
        This section summarizes patterns in mobile prices,
        customer ratings, and brand performance.
        """
    )

    # -----------------------------------------------------
    # Insight 1: Highest average-rated brand
    # -----------------------------------------------------

    st.subheader("1. Brand Rating Performance")

    brand_performance = (
        df.groupby("brand", as_index=False)
        .agg(
            average_rating=("rating", "mean"),
            review_count=("rating", "count"),
            average_price=("price_usd", "mean")
        )
    )

    # Show brands with at least 5 review records
    eligible_brands = brand_performance[
        brand_performance["review_count"] >= 5
    ]

    if not eligible_brands.empty:

        best_brand = eligible_brands.loc[
            eligible_brands["average_rating"].idxmax()
        ]

        st.write(
            f"Among brands with at least 5 review records, "
            f"**{best_brand['brand']}** has the highest average rating "
            f"of **{best_brand['average_rating']:.2f}/5**."
        )

    st.dataframe(
        brand_performance.sort_values(
            "average_rating",
            ascending=False
        ).round(2),
        use_container_width=True
    )

    # -----------------------------------------------------
    # Insight 2: Most expensive models
    # -----------------------------------------------------

    st.subheader("2. Highest-Priced Models")

    model_prices = (
        df.groupby(["brand", "model"], as_index=False)
        .agg(
            average_price=("price_usd", "mean"),
            average_rating=("rating", "mean")
        )
        .sort_values("average_price", ascending=False)
    )

    st.dataframe(
        model_prices.head(10).round(2),
        use_container_width=True
    )

    # -----------------------------------------------------
    # Insight 3: Price and rating relationship
    # -----------------------------------------------------

    st.subheader("3. Price and Customer Rating Relationship")

    valid_data = df[["price_usd", "rating"]].dropna()

    if (
        len(valid_data) >= 2
        and valid_data["price_usd"].nunique() > 1
        and valid_data["rating"].nunique() > 1
    ):

        correlation = valid_data[
            "price_usd"
        ].corr(valid_data["rating"])

        st.metric(
            "Price-Rating Correlation",
            f"{correlation:.3f}"
        )

        if correlation > 0.3:

            st.write(
                "The dataset shows a positive association between "
                "price and rating."
            )

        elif correlation < -0.3:

            st.write(
                "The dataset shows a negative association between "
                "price and rating."
            )

        else:

            st.write(
                "The dataset shows a relatively weak linear association "
                "between price and rating."
            )

        st.caption(
            "Correlation does not prove that price causes higher "
            "or lower customer ratings."
        )

    # -----------------------------------------------------
    # Insight 4: Cluster comparison
    # -----------------------------------------------------

    if "cluster" in df.columns:

        st.subheader("4. Product Segment Comparison")

        segment_summary = (
            df.groupby("cluster", as_index=False)
            .agg(
                average_price=("price_usd", "mean"),
                average_rating=("rating", "mean"),
                record_count=("model", "count"),
                distinct_models=("model", "nunique")
            )
            .round(2)
        )

        st.dataframe(
            segment_summary,
            use_container_width=True
        )

        st.write(
            """
            Business applications:

            - Identify product groups with relatively high ratings.
            - Compare expensive products with their customer ratings.
            - Explore lower-priced models that receive positive ratings.
            - Use cluster characteristics to support product positioning.
            - Use similarity-based recommendations to help customers
              explore alternative mobile models.
            """
        )

    st.divider()

    st.caption(
        "All findings are calculated from the dataset loaded by this app. "
        "They describe observed patterns and should not be interpreted "
        "as proof of causation."
    )