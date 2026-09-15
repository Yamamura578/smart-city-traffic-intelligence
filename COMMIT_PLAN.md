# Suggested commit sequence

The rubric grades incremental, descriptively named commits. Commit after each
working step, not in one dump at the end.

```
chore: initialise repository structure and requirements
chore: add shared logging configuration and project constants
docs: add README with structure, run instructions and logging documentation
data: add raw Metro Interstate traffic dataset

feat(part1): load dataset into SQLite and verify row counts
feat(part1): add yearly traffic trend queries
feat(part1): add holiday temperature comparison queries
feat(part1): add descriptive statistics and correlation analysis
feat(part1): add probability and conditional probability analysis
feat(part1): add Power BI dashboard and Power Query preparation steps
docs(part1): add data analytics insights report

feat(part2): implement raw CSV loading with exception handling and logging
feat(part2): add schema validation before cleaning
feat(part2): parse datetime and remove duplicate records
feat(part2): standardise categoricals and forward-fill holiday column
feat(part2): detect and median-impute impossible temperature and rainfall values
feat(part2): add time and cyclical features
feat(part2): add weather encodings and derived indicators
feat(part2): add scaled numeric features and quartile congestion target
feat(part2): add hourly, weather and temperature visualisations
feat(part2): add CLI application with lookup, peak and advise commands
docs(part2): add methodology report and sample pipeline log

feat(part3): add proxy accident-risk label with documented derivation
feat(part3): add logistic regression and random forest classifiers
feat(part3): add linear and gradient boosting regressors with MAE/R2 comparison
feat(part3): add k-means clustering of traffic conditions
feat(part3): add association rule mining on discretised conditions
feat(part3): add LSTM demand model
feat(part3): add SHAP explainability on tree-based surrogate
feat(part3): add MLflow experiment tracking
feat(part3): add FastAPI deployment mock-up
feat(part3): add drift monitoring and PASS/ALERT reporting
feat(part3): add travel-timing recommendation engine
docs(part3): add final capstone report and bias and fairness report
```
