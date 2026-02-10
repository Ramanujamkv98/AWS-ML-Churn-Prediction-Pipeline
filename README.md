AWS Churn Prediction Pipeline — End-to-End ML + Analytics Project

This project was my attempt to build a real-world churn prediction system from scratch — not just a model in a notebook, but a full pipeline covering data storage, processing, training, evaluation, dashboarding, and live inference on AWS. My goal was to understand how ML systems behave in production, not just in theory.

Project Objective

I aimed to predict the probability of a user churning within 14 days by combining:

Behavioral usage metrics

Experience / friction signals

Engagement trends over time

I then exposed these predictions through a live Streamlit web app running on an EC2 instance, so it felt like a real product rather than an offline experiment.

Architecture & Tools I Used

Amazon S3 — data lake and model artifact storage

Athena — quick SQL validation and leakage checks

EC2 — compute for deployment and hosting Streamlit

IAM Roles — secure S3 access without hard-coded keys

Power BI — exploratory data analysis dashboards

Streamlit — real-time prediction interface

Data Flow & Layers I Built

Raw Layer (S3): original dataset with timestamps and IDs, minimal cleaning

Curated Layer (S3): removed leakage columns, validated labels, prepared time splits

Processed Layer (S3): feature engineering, one-hot encoding, train/validation/test parquet files

Models Layer (S3): stored Logistic Regression pipeline, XGBoost model, feature schema, and deployment artifacts

What I Learned from EDA

While exploring the data in Power BI, I noticed churn was more tied to declining engagement trends over time than single-day low usage. Users with fewer active days and dropping token counts churned more. Latency and errors mattered, but experienced users sometimes tolerated friction if they still found value. Monthly churn gradually declined, hinting at product adaptation.

Modeling Approach

I trained two models:

Logistic Regression for interpretability and coefficient insights

XGBoost to capture non-linear interactions and slightly improve recall

Accuracy looked high because of class imbalance, but recall was realistically low — a common churn challenge. Using both models gave me complementary perspectives instead of relying on one algorithm.

Deployment

I deployed a Streamlit UI on an EC2 t3.micro instance with public IP access and IAM-based S3 permissions. Users can enter feature values, choose a model, adjust thresholds, and instantly see churn probability — turning analysis into something interactive.

Roadblocks & Key Learnings

Reading Parquet prefixes vs. files in S3

Handling class imbalance and misleading accuracy

Keeping training and inference feature schemas aligned

IAM permission setup for secure EC2–S3 communication

Separating intuitive EDA patterns from model-weighted importance

Realizing churn is often gradual and behavioral, not abrupt

Final Reflection

This project taught me that engagement trends and user maturity often outweigh isolated performance metrics in churn prediction. EDA gave me intuition; models quantified predictive strength. Sometimes they disagreed, which reminded me that descriptive and predictive analytics are complementary. Comparing multiple models increased my confidence and depth of understanding.

Further improvements can be adding richer user history, resampling techniques, and better threshold optimization to capture minority churn cases more effectively.
