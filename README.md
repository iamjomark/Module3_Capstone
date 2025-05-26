# Travel Insurance Claim Prediction

## Project Overview

This project aims to build a machine learning model that predicts whether a travel insurance policyholder will file a claim. This prediction helps the insurance company proactively manage risk, reduce financial losses, and optimize premium pricing strategies.

## Business Context

Travel insurance providers face high uncertainty in claim outcomes. With only a small percentage of policyholders filing claims (~2%), identifying high-risk customers before a claim is made is difficult but crucial. This project tackles that challenge using data-driven risk classification and predictive modeling.

## Stakeholders

- **Primary Stakeholder**: The **insurance company**, which needs to manage claim-related expenses and price products more effectively.
- **Secondary Stakeholder**: The **customers**, who are impacted by pricing, claim approvals, and risk profiling.

## Problem Statement

The insurance provider struggles to **accurately identify high-risk policies**, resulting in:
- Missed claims (false negatives) that cause **large financial payouts**.
- Incorrectly flagged customers (false positives) that **increase admin workload**.

This issue affects cost management, resource allocation, and customer trust.

## Why It Matters

- **Specific Impact**:
  - Each missed claim can cost around **$1,000**.
  - Each false positive adds an estimated **$50** in processing costs.
- **Measurable Risk**: High false negative or false positive rates directly translate to **monetary losses and operational inefficiency**.

## Project Goals

This project aims to:
- Build a classification model (e.g., LightGBM) to **predict the likelihood of a claim**.
- Focus on **maximizing recall and F2-score**, prioritizing missed claim reduction.
- Estimate the **financial impact** of misclassifications using realistic assumptions.
- Generate **business insights** (e.g., risk drivers, fraud-prone agencies, product risks) using model explainability (SHAP values).

### Target Metrics
- **Recall ≥ 60%** to reduce missed claims.
- **F2 Score Optimization** to prioritize recall while considering precision.
- **Total Financial Impact**: Minimize combined cost from FN and FP cases.

## Files
- Data: data_travel_insurance.csv
- Glossary: Travel Insurance.docx
- Jupyter Notebook: M3_CS_Travel Insurance_Jonathan Mark H.ipynb
- Saved Models Folder (with results): saved_models


