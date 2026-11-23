# Retail Customer Churn Prediction

## 1. Project Overview

This project focuses on analysing customer behaviour and identifying factors associated with customer churn in a retail business.

The objective is to conduct exploratory data analysis (EDA), identify meaningful patterns in customer behaviour, and prepare the dataset for developing a machine learning classification model to predict whether a customer is likely to churn.

**Problem Type:** Binary Classification  
**Target Variable:** `customer_churn`

## 2. Dataset Overview

The dataset contains 1,000 customer records and 7 columns, comprising six input features and one target variable.

| Feature | Description |
|---|---|
| `monthly_spend` | Customer's monthly spending |
| `number_of_store_visits_per_month` | Number of store visits per month |
| `customer_satisfaction_score` | Customer satisfaction rating |
| `number_of_complaints` | Number of customer complaints |
| `membership_tier` | Customer membership category |
| `discount_usage_rate` | Rate of discount usage |
| `customer_churn` | Customer churn status (target variable) |

The target variable consists of two classes:
- **Not Churned:** 850 customers (85%)
- **Churned:** 150 customers (15%)

The dataset therefore has an imbalanced target distribution, with non-churned customers forming the majority class.

## 3. Exploratory Data Analysis (EDA)

EDA was conducted to understand the dataset's structure, distributions, relationships between variables, and potential data-quality issues.

### 3.1 Data Quality Assessment

Initial checks identified:
- 1,000 records and 7 columns.
- No missing values.
- No duplicate records.
- An imbalanced target variable, with a 15% churn rate.

Several unusual patterns were also identified:
- `number_of_store_visits_per_month` and `number_of_complaints` contain negative decimal values. Their distributions are centred approximately around zero, suggesting that they may have been standardised. However, the original transformation process is unknown.
- Membership categories are Basic, Premium and VIP.
- Monthly spending is unexpectedly higher among Basic members compared with Premium and VIP members.

These observations should be considered during preprocessing and model development. The original values and target labels have been retained pending further clarification of the dataset's underlying data-generation process.

### 3.2 Key Numerical Findings

**Monthly Spend**
- Right-skewed distribution with several upper-end outliers.
- Churned customers have higher median spending than non-churned customers.
- This pattern is also observed across all three membership tiers.

**Customer Satisfaction**
- Scores are generally concentrated in the middle-to-higher range.
- Churned customers have a higher median satisfaction score than non-churned customers.
- This is an unexpected pattern that warrants further investigation.

**Store Visits**
- Distribution is approximately normal.
- Churned customers have a slightly higher median number of visits.
- Both customer groups show considerable overlap.

**Customer Complaints**
- Distribution is approximately normal and centred around zero.
- Churned customers have a lower median value than non-churned customers.
- Negative values suggest the feature may have undergone standardisation.

**Discount Usage**
- Right-skewed distribution, with most customers recording relatively low-to-moderate usage.
- Churned customers have a slightly higher median discount usage rate.
- The two groups show considerable overlap.

### 3.3 Membership Tier and Churn

Churn rates differ across membership tiers:

| Membership Tier | Customers | Churned | Churn Rate |
|---|---:|---:|---:|
| Basic | 500 | 97 | 19.4% |
| Premium | 300 | 39 | 13.0% |
| VIP | 200 | 14 | 7.0% |

Basic members have the highest churn rate, while VIP members have the lowest.

This suggests that membership tier may be associated with customer retention. However, further analysis would be required to establish the underlying reasons.

### 3.4 Correlation Analysis

The correlation analysis identified the following relationships:

- Customer satisfaction and complaints: moderate negative correlation (approximately -0.59).
- Monthly spend and discount usage: weak-to-moderate positive correlation (approximately 0.31).
- Monthly spend and store visits: very weak positive correlation (approximately 0.17).

These findings indicate that some features share relationships, although correlation does not imply causation.

## 4. Statistical Testing

Statistical tests were conducted to assess whether observed differences between churned and non-churned customers were statistically significant.

- **Mann–Whitney U tests:** Used to compare numerical feature distributions between the two customer groups.
- **Chi-square test of independence:** Used to examine the relationship between membership tier and customer churn.
- **Holm correction:** Applied to adjust p-values for multiple hypothesis testing.
- **Significance level:** 0.05.

### 4.1 Statistical Test Results

| Feature | Test | Adjusted p-value | Significant? |
|---|---|---:|---|
| Monthly Spend | Mann–Whitney U | 2.43 × 10⁻¹⁵ | Yes |
| Store Visits | Mann–Whitney U | 0.0685 | No |
| Complaints | Mann–Whitney U | 5.83 × 10⁻¹⁷ | Yes |
| Satisfaction | Mann–Whitney U | 1.00 × 10⁻¹⁹ | Yes |
| Discount Usage | Mann–Whitney U | 0.2265 | No |
| Membership Tier | Chi-square | 0.000278 | Yes |

### 4.2 Interpretation

After Holm correction, four of the six tested features showed statistically significant associations or distributional differences:

- Monthly spending
- Number of complaints
- Customer satisfaction
- Membership tier

Store visits and discount usage did not show statistically significant differences at the 0.05 significance level after correction.

These results indicate differences in the observed data but do not establish causation or guarantee predictive usefulness.

## 5. Key Business Insights

The EDA revealed several observations that may be relevant to understanding customer churn:

1. **Basic members have the highest churn rate.** Their churn rate of 19.4% is considerably higher than Premium (13.0%) and VIP (7.0%) members.

2. **Higher spending is associated with churn in this dataset.** Churned customers have higher average and median monthly spending across all membership tiers.

3. **Higher satisfaction scores are observed among churned customers.** This contradicts conventional expectations and should be investigated rather than assumed to represent an error.

4. **Customer complaints show an unexpected pattern.** Churned customers have lower median values on the complaints feature, although its negative decimal values suggest that the feature may have been transformed.

5. **Some features have substantial overlap between churn groups.** This suggests that individual features may not be sufficient to distinguish churned customers from non-churned customers.

## 6. Data Considerations and Limitations

Several limitations should be considered before proceeding with model development:

- The dataset is synthetic, and the underlying data-generation process is not documented.
- The negative decimal values in store visits and complaints may represent standardised values rather than raw counts.
- Customer spending patterns across membership tiers are counterintuitive.
- Churned customers display higher satisfaction and spending in the observed data, which differs from conventional business expectations.
- The target distribution is imbalanced, with only 15% of customers classified as churned.

The existing churn labels will be retained. There is insufficient evidence to conclude that the labels have been reversed or incorrectly assigned.

## 7. Implications for Model Development

The EDA findings will inform the subsequent preprocessing and modelling stages.

Key considerations include:

- Preserve the original target labels.
- Investigate unusual numerical values before deciding on any transformations.
- Encode the categorical membership feature appropriately.
- Use stratified data splitting to preserve the churn distribution across datasets.
- Evaluate model performance using metrics beyond accuracy, particularly precision, recall and F1-score, given the imbalanced target distribution.
- Ensure preprocessing steps are fitted using training data only to avoid data leakage.

## 8. Tools and Libraries

The analysis was conducted using Python and the following libraries:

- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical operations.
- **Matplotlib:** Data visualisation.
- **Seaborn:** Statistical visualisation.
- **SciPy:** Statistical hypothesis testing.
- **Statsmodels:** Multiple hypothesis testing and p-value adjustment.
- **Jupyter Notebook:** Interactive analysis and documentation.

## 9. Conclusion

The exploratory analysis identified several potentially meaningful relationships between customer characteristics and churn, particularly in monthly spending, customer satisfaction, complaints and membership tier.

However, some observed patterns contradict conventional retail expectations. These findings highlight the importance of understanding the dataset's limitations and avoiding assumptions based solely on business intuition.

The results provide a foundation for the next stage of the project: data preprocessing, feature engineering and machine learning model development.