# Credit Risk Model Project

## Project Structure
```text
 credit-risk-model/ 
 ├── .github/workflows/ci.yml      # CI/CD pipeline 
 ├── data/                          # add to .gitignore 
 │   ├── raw/                       # Raw data 
 │   └── processed/                 # Processed data for training 
 ├── notebooks/ 
 │   └── eda.ipynb                  # Exploratory analysis 
 ├── src/ 
 │   ├── __init__.py 
 │   ├── data_processing.py         # Feature engineering 
 │   ├── train.py                   # Model training 
 │   ├── predict.py                 # Inference 
 │   └── api/ 
 │       ├── main.py                # FastAPI application 
 │       └── pydantic_models.py     # Request/response schemas 
 ├── tests/ 
 │   └── test_data_processing.py    # Unit tests 
 ├── Dockerfile 
 ├── docker-compose.yml 
 ├── requirements.txt 
 ├── .gitignore 
 └── README.md 
```

## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability
The Basel II Accord's emphasis on risk measurement—specifically through the **Internal Ratings-Based (IRB) approach**—demands that models used for calculating regulatory capital be both interpretable and thoroughly documented. This is because these models directly influence the minimum capital requirements of a bank. 
- **Regulatory Oversight:** Regulators require transparency to ensure that banks are not underestimating risk to reduce capital holdings.
- **Model Validation:** An interpretable model allows for robust "backtesting" and "stress testing," ensuring the model performs as expected during economic downturns.
- **Decision Support:** Documentation ensures that the model's logic is consistent with the bank's risk appetite and can be audited by internal and external parties.

### 2. Proxy Variables and Business Risks
In credit risk modeling, a direct "default" label (e.g., total loss) may be rare or take years to materialize. A **proxy variable** (such as "90 days past due" or "unlikely to pay") is necessary to provide enough data points for statistical modeling.
- **Why it's necessary:** Proxies provide a timely and sufficient signal for the model to learn the characteristics of risky borrowers without waiting for a multi-year default cycle to complete.
- **Business Risks:**
    - **Model Misalignment:** The proxy may not perfectly correlate with actual financial loss, leading to "false alarms" or missed defaults.
    - **Adverse Selection:** If the proxy is too strict, the bank may lose profitable customers; if too lenient, it may accumulate toxic debt.
    - **Regulatory Non-compliance:** If the proxy is not defensible, regulators may reject the model, forcing the bank to use more expensive standardized capital weights.

### 3. Trade-offs: Simple vs. High-Performance Models
In a regulated financial context, the choice between a simple model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) involves significant trade-offs:

| Feature | Logistic Regression + WoE | Gradient Boosting (e.g., XGBoost) |
|---------|---------------------------|-----------------------------------|
| **Interpretability** | **High:** Easy to explain the impact of each variable (scorecard format). | **Low:** Often viewed as a "black box" regarding individual decisions. |
| **Performance** | **Moderate:** Limited to linear relationships and manual feature interactions. | **High:** Captures complex, non-linear patterns and interactions. |
| **Regulatory Acceptance** | **High:** Gold standard for regulatory compliance and auditability. | **Increasing:** Requires complex "Explainable AI" (XAI) tools for approval. |
| **Stability** | **High:** Less prone to overfitting and more stable across different economic cycles. | **Moderate:** Can be sensitive to noise and requires careful tuning. |
| **Implementation** | **Simple:** Can be implemented as a simple lookup table or scoring logic. | **Complex:** Requires more computational resources and infrastructure. |

In many regulated environments, banks often prefer the **Logistic Regression with WoE** approach for its transparency and ease of providing "adverse action notices" to customers, while using GBMs as a benchmark for potential performance gains.
