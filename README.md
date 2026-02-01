AWS Churn Prediction Pipeline: End-to-End ML + Analytics Project
This project is an attempt to build a real-world style churn prediction system from scratch not just a model, but a full pipeline that covers data storage, processing, model training, evaluation, dashboarding, and live inference deployment on AWS.
The goal was to understand how machine learning systems behave in production environments, not only in notebooks.

Project Objective:
Predict the probability of a user churning within 14 days by combining:
•	Behavioral usage metrics
•	Experience / friction signals
•	Engagement trends over time
and then exposing the prediction through a live Streamlit web interface running on an EC2 instance.

Architecture Overview:
AWS Services Used
•	Amazon S3 — Data lake + model artifact storage
•	Athena — Quick SQL validation and leakage checks
•	EC2 — Compute for deployment and Streamlit UI
•	IAM Roles — Secure S3 access without hard-coded keys
•	Power BI — Exploratory data analysis dashboard
•	Streamlit — Real-time prediction interface

Data Flow & Layers:
Raw Layer (S3)
Original synthetic dataset with timestamps, IDs, and minimal cleaning.
Curated Layer (S3)
•	Leakage column removed
•	Labels validated
•	Prepared for time-based splitting
Processed Layer (S3)
•	Feature engineering applied
•	One-hot encoding
•	Time-aware Train / Validation / Test splits saved as Parquet.
Models Layer (S3)
Stored trained artifacts:
•	Logistic Regression pipeline
•	XGBoost model
•	Feature schema
•	Deployment shortcut model

Exploratory Data Analysis (Power BI):
EDA revealed that churn was more strongly associated with declining engagement trends (sessions and tokens decreasing over time) rather than single-day low usage.
Low token users and fewer active days showed higher churn.
Interestingly, higher latency and error rates increased risk but did not always dominate, suggesting that experienced users may tolerate friction if they still find value.
Monthly churn showed a gradual decline, hinting at possible product adaptation effects.

Modeling Approach
Two models were trained sequentially:
•	Logistic Regression — Provided clear coefficient-based interpretability.
•	XGBoost — Captured non-linear interactions and slightly improved recall.
Accuracy was high due to class imbalance, but recall remained low a realistic challenge in churn problems.
Combining both models added a broader perspective, as linear and tree-based methods highlighted patterns from different angles.

Deployment:
A Streamlit UI was deployed on an EC2 t3.micro instance with:
•	Public IP access
•	Security group rules for HTTP/Custom TCP
•	IAM role permissions allowing S3 read access
Users can input feature values, select a model, adjust the decision threshold, and receive real-time churn probability predictions.

Key Roadblocks & Learnings:
•	Reading Parquet prefixes vs files in S3
•	Handling class imbalance and misleading accuracy
•	Aligning feature schemas between training and inference
•	IAM permissions (GetObject, ListBucket) for secure EC2–S3 communication
•	Distinguishing EDA intuition vs model-weighted influence
•	Understanding that churn is often gradual and behavioral, not abrupt.

Final Reflection:
This project highlighted that engagement trends and user maturity often outweigh isolated performance metrics in churn prediction.
EDA provided intuitive behavioral patterns, while models quantified predictive strength and interaction effects.
Some outcomes differed between visual analysis and model importance, reinforcing that descriptive and predictive analytics serve complementary roles.
Comparing multiple models added confidence and depth, emphasizing that no single algorithm tells the complete story. Future improvements would include richer user history, resampling techniques, and threshold optimization to better capture minority churn events.

