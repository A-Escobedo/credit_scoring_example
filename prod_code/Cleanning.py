# Import libraries
import pandas as pd

# Read datasets
general_info = pd.read_csv("datasets\general_info.csv")
internal_payments = pd.read_csv("datasets\internal_payments.csv")
credit_reports = pd.read_csv("datasets\credit_reports.csv")
external_features = pd.read_csv("datasets\external_features.csv")

def get_train_data(general_info, internal_payments, credit_reports):
    """Get data before limit_date"""
    # clean general info
    clean_general_info = general_info[["application_id", "target"]]

    # clean_internal_payments
    clean_internal_payments = internal_payments.query("payment_date<limit_date")

    # clean_credit_reports
    clean_credit_reports = credit_reports.query("report_date<=limit_date")

    return clean_general_info, clean_internal_payments, clean_credit_reports

clean_general_info, clean_internal_payments, clean_credit_reports = get_train_data(general_info=general_info, internal_payments=internal_payments, credit_reports=credit_reports)