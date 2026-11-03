Trading Risk Monitor

Detects anomalous trading behaviour and classifies trades as risky or normal, using a real retail trade book.

What it does
- Calculates five trade features and a rule-based risk score
- Detects unusual trades with Isolation Forest
- Compares five classifiers with cross-validation; Gradient Boosting selected (test F1 0.969)
- Local Streamlit dashboard for filtering trades and inspecting warning signs

Limitations
The risk label is built from the same features the models learn, so scores show how well the models reproduce the rule. They do not show prediction of real trading losses.

Run it
pip install -r requirements.txt
streamlit run app.py

Upload your own Kite trade book in the sidebar.
