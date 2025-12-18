#Fraud Detection Dashboard
Fraud Detection Dashboard takes input from the user and predicts whether the transaction is Fraud,Legitimate or Suspicious based upon the rule-based logic.The model is trained using Random Forest with ~0.99 Accuracy.Users can submit their transactional data ,instantly view their risk score,track transaction data history and analyse trends through visual dashboard.
##Features
* Users can enter transaction details and instantly view the predicted fraud category along with a risk score.
* All transactions and predictions are automatically stored in an Excel file for auditing and analysis
* Users can view past transactions and their predictions directly within the dashboard.
* Combines Random Forest machine learning with rule-based logic to improve detection accuracy and handle edge cases.
* Built-in charts to analyze fraud distribution and risk patterns over time.

1.Fraud Detection Logic
* Trained dataset on Random Forest Classifier (rf_fraud_model.pkl) which outputs base fraud probability.
2. Feature Engineering
