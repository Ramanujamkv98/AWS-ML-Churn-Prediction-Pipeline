# AWS Churn Prediction Pipeline + Retention Decisioning


## Project Index

### Phase 1: End-to-End ML Pipeline

Built an AWS-based churn prediction pipeline for an LLM-style product, covering data storage, SQL validation, feature engineering, model training, dashboarding, and live inference.

Key components:

- S3 for raw, curated, processed, and model artifact layers
- Athena for SQL validation and leakage checks
- Logistic Regression and XGBoost for churn modeling
- Power BI for behavioral EDA
- EC2 and Streamlit for live inference
- IAM roles for secure EC2-S3 access

Main learning: churn prediction is not just about fitting a model. Cloud data flow, permissions, feature schema consistency, and deployment design matter just as much as model metrics.

### Phase 2: Retention Decision System

Extended the churn project beyond prediction into retention decisioning. This phase asks: once the model identifies high-risk users, should the business intervene?

Key components:

- XGBoost churn scoring on a synthetic 1M-user learning-app dataset
- 3.1% churn rate to reflect class imbalance
- Threshold tradeoff analysis using precision, recall, false positives, and LTV
- SHAP analysis to explain churn-risk drivers
- Simulated holdout experiment among users flagged at a 0.70 threshold
- Retention discount test with SRM checks and two-proportion z-test
- Business impact analysis using retained users, net LTV, discount cost, ROI, and unsubscribe guardrails

Key result: the model identified a high-risk segment with 31% precision compared with a 3.1% overall churn rate. The simulated discount reduced churn from 29.6% to 27.2%, but the economics were negative after discount cost and unsubscribe-risk adjustment. Recommendation: do not scale the discount as-is; test lower-cost nudges and better targeting.

Main learning: prediction helps decide where to act, but experimentation and business economics decide whether the action is worth scaling.

---

## Phase 1: End-to-End ML Pipeline

This project was my attempt to build a real-world churn prediction system from scratch, not just a model in a notebook, but a full pipeline covering data storage, processing, training, evaluation, dashboarding, and live inference on AWS. My goal was to understand how ML systems behave in production, not just in theory.

## Project Objective

I aimed to predict the probability of a user churning within 14 days by combining:

1. Behavioral usage metrics
2. Experience / friction signals
3. Engagement trends over time

I then exposed these predictions through a live Streamlit web app running on an EC2 instance, so it felt like a real product rather than an offline experiment.

## Architecture & Tools I Used

- Amazon S3 — data lake and model artifact storage
- Athena — quick SQL validation and leakage checks
- EC2 — compute for deployment and hosting Streamlit
- IAM Roles — secure S3 access without hard-coded keys
- Power BI — exploratory data analysis dashboards
- Streamlit — real-time prediction interface

## Data Flow & Layers I Built

- Raw Layer (S3): original dataset with timestamps and IDs, minimal cleaning
- Curated Layer (S3): removed leakage columns, validated labels, prepared time splits
- Processed Layer (S3): feature engineering, one-hot encoding, train/validation/test parquet files
- Models Layer (S3): stored Logistic Regression pipeline, XGBoost model, feature schema, and deployment artifacts

## What I Learned from EDA

![image (10)](https://github.com/user-attachments/assets/69b917ec-8397-48e5-b7c4-06febe17b6cb)

Identified an 800+ tokens/session threshold where only 7.8% of users reached this level, but churn was 62% lower.

While exploring the data in Power BI, I noticed churn was more tied to declining engagement trends over time than single-day low usage. Users with fewer active days and dropping token counts churned more. Latency and errors mattered, but experienced users sometimes tolerated friction if they still found value. Monthly churn gradually declined, hinting at product adaptation.

## Modeling Approach

I trained two models:

- Logistic Regression for interpretability and coefficient insights
- XGBoost to capture non-linear interactions

Accuracy looked high because of class imbalance, but recall was realistically low, a common churn challenge. Using both models gave me complementary perspectives instead of relying on one algorithm.

## Deployment

I deployed a Streamlit UI on an EC2 t3.micro instance with public IP access and IAM-based S3 permissions. Users can enter feature values, choose a model, adjust thresholds, and instantly see churn probability, turning analysis into something interactive.

## Roadblocks & Key Learnings

- Reading Parquet prefixes vs. files in S3
- Handling class imbalance and misleading accuracy
- Keeping training and inference feature schemas aligned
- IAM permission setup for secure EC2-S3 communication
- Separating intuitive EDA patterns from model-weighted importance
- Realizing churn is often gradual and behavioral, not abrupt

## Phase 2: Retention Decision System

The first phase answered: who is likely to churn?

This extension answers: should we intervene, and is the intervention worth scaling?

I built a synthetic 1M-user learning-app case study with approximately 3.1% 14-day churn. The model used behavioral features such as active days, lessons completed, session duration, days since last active, and support tickets.

At a 0.70 threshold, the model flagged 5,518 users as high risk. This segment had a 31% observed churn rate, compared with 3.1% overall churn, showing that the model concentrated risk meaningfully.

I then simulated a retention experiment only on this high-risk segment. Users were randomly split into control and treatment groups. The treatment group received an 8 retention discount.

The intervention reduced churn from 29.6% to 27.2%, a 2.4 percentage-point absolute reduction and about 8% relative reduction. The result was statistically significant at the 5% level.

However, the business case did not justify scaling. The intervention retained an estimated 67 incremental users and created about 12.1K in contribution LTV, but the discount cost was about 22.0K. Unsubscribe rates also increased from 2.4% to 3.7%, creating an additional estimated future marketing-reach cost. Final guardrail-adjusted net value was about -10.3K.

## Phase 2 Recommendation

The model worked, and the intervention reduced churn, but the discount economics were negative. I would not scale the discount as-is. The next test should explore lower-cost interventions such as:

- in-app nudges
- personalized progress reminders
- smaller discounts
- frequency caps
- non-discount lifecycle messaging
- improved targeting using uplift modeling

## Final Reflection

This project helped me understand the difference between model performance, intervention impact, and business decision quality.

Prediction helps decide where to act. Experimentation helps decide whether the action works. Business economics and guardrails decide whether it should scale.
