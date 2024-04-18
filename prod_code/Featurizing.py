# Import libraries
import numpy as np
import pandas as pd
from Cleanning import clean_general_info, clean_internal_payments, clean_credit_reports

########## auxiliary functions ##########
########## Internal payments ##########
def get_aux_cols_internal_payments(internal_payments_df):
    """Create auxiliar columns from internal payments."""
    # Convert date columns to correct format
    internal_payments_df["effective_maturity_date"] = pd.to_datetime(internal_payments_df["effective_maturity_date"]).dt.date
    internal_payments_df["limit_date"] = pd.to_datetime(internal_payments_df["limit_date"]).dt.date
    internal_payments_df["payment_date"] = pd.to_datetime(internal_payments_df["payment_date"]).dt.date
    internal_payments_df["completed_date"] = pd.to_datetime(internal_payments_df["completed_date"]).dt.date
    
    # Create column related to fully_paid contracts
    internal_payments_df["is_finished_contract"] = np.where(internal_payments_df["effective_maturity_date"]<= internal_payments_df["limit_date"], 1, 0)
    # Create dummy: is_late_payment
    internal_payments_df["is_late_payment"] = np.where(internal_payments_df["payment_date"]<internal_payments_df["completed_date"], 1, 0)
    return None

def get_internal_features(internal_payments_df):
    """Calculate credit report features."""
    # Set dataframe
    INTERNAL_INPUT = internal_payments_df.copy()
    # Create auxiliar columns
    get_aux_cols_internal_payments(internal_payments_df=INTERNAL_INPUT)

    # Create internal features
    internal_features = INTERNAL_INPUT.groupby("application_id").agg(
        num_prev_contracts=pd.NamedAgg(column="is_finished_contract", aggfunc="sum"),
        avg_notional=pd.NamedAgg(column="notional", aggfunc="mean"),
        pct_late_payments=pd.NamedAgg(column="is_late_payment", aggfunc="mean"),
        internal_credit_payments = pd.NamedAgg(column="payment_number", aggfunc="count")
    )

    return internal_features

########## credit reports ##########
def get_aux_cols_credit_reports(credit_reports_df):
    """Create auxiliar columns from credit reports."""
    # Create new columns related to open and closing_date
    credit_reports_df["is_open_account"] = np.where(credit_reports_df["account_closing_date"].isna(), 1, 0)
    credit_reports_df["is_closed_account"] = np.where(credit_reports_df["account_closing_date"].isna(), 0, 1)
    return None

# Unify categories
def unify_institution(credit_reports_df):
    """Unify categories from a specific credit reports dataframe"""
    credit_reports_df.loc[credit_reports_df['institution'].isin(['FONDOS Y FIDEIC', 'FONDOS Y FIDEICO','FONDOS Y FIDEICOMISOS']), 'institution'] = 'FONDOS Y FIDEICOMISOS'
    credit_reports_df.loc[credit_reports_df['institution'].isin(['ARRENDADORA', 'ARRENDADORAS FINANCIERAS', 'ARRENDADORAS NO FINANCIERAS', 'ARRENDAMIENTO']), 'institution'] = 'ARRENDADORA'
    credit_reports_df.loc[credit_reports_df['institution'].isin(['BANCO', 'BANCOS']), 'institution'] = 'BANCO'
    credit_reports_df.loc[credit_reports_df['institution'].isin(['COMPANIA DE FINANCIAMIENTO AUTOMOTRIZ','COMPANIA DE FINANCIAMIENTO DE MOTOCICLET']), 'institution'] = 'COMPANIA DE FINANCIAMIENTO AUTOMOTRIZ'
    credit_reports_df.loc[credit_reports_df['institution'].isin(['COMPANIA DE FINANCIAMIENTO AUTOMOTRIZ','COMPANIA DE FINANCIAMIENTO DE MOTOCICLET']), 'institution'] = 'COMPANIA DE FINANCIAMIENTO AUTOMOTRIZ'
    credit_reports_df.loc[credit_reports_df['institution'].isin(['GOBIERNO','GUBERNAMENTALES']), 'institution'] = 'GOBIERNO'
    credit_reports_df.loc[credit_reports_df['institution'].isin(['MICROFINANCIERA', 'OTRAS FINANCIERA']), 'institution'] = 'OTRAS FINANCIERA'
    credit_reports_df.loc[credit_reports_df['institution'].isin(['TELEFONIA CELULAR','TELEFONIA LOCAL Y DE LARGA DISTANCIA']), 'institution'] = 'TELEFONIA'
    credit_reports_df.loc[credit_reports_df['institution'].isin(['TIENDA COMERCIAL','TIENDA DE AUTOSERVICIO', 'TIENDA DE ROPA', 'TIENDA DEPARTAMENTAL']), 'institution'] = 'TIENDA'
    return None

# Get credit features
def get_credit_features(credit_reports_df):
    """Calculate credit report features."""
    # Set dataframe
    EXTERNAL_INPUT = credit_reports_df.copy()

    # Create auxiliar columns
    get_aux_cols_credit_reports(credit_reports_df=EXTERNAL_INPUT)

    # Unify institution categories
    unify_institution(credit_reports_df=EXTERNAL_INPUT)

    # Define categorical features
    columns_to_encode = ["institution", "account_type", "credit_type"]
    original_columns = EXTERNAL_INPUT.columns

    # Get dummy columns
    credit_reports_dummies = pd.get_dummies(EXTERNAL_INPUT, columns=columns_to_encode, drop_first=True)
    all_columns = credit_reports_dummies.columns
    dummy_columns = [col for col in all_columns if col not in original_columns]
    # Add application_id
    dummy_columns.append("application_id")

    # Create dummy features
    credit_reports_dummy_features = credit_reports_dummies[dummy_columns].groupby("application_id").agg("sum").astype(bool)
    # Create numerical features
    credit_reports_numerical_features = EXTERNAL_INPUT.groupby("application_id").agg(
        open_accounts=pd.NamedAgg(column="is_open_account", aggfunc="sum"),
        closed_accounts=pd.NamedAgg(column="is_closed_account", aggfunc="sum"),
        max_credit_amount=pd.NamedAgg(column="maximum_credit_amount", aggfunc="max"),
        current_balance = pd.NamedAgg(column="current_balance", aggfunc="sum"),
        past_due_balance = pd.NamedAgg(column="past_due_balance", aggfunc="sum"),
        total_credit_payments = pd.NamedAgg(column="total_credit_payments", aggfunc="sum"),
        worst_delinquency_past_due_balance = pd.NamedAgg(column="worst_delinquency_past_due_balance", aggfunc="max"),
        credit_limit = pd.NamedAgg(column="credit_limit", aggfunc="sum")
    )

    # Create derived features
    credit_reports_numerical_features["past_due_ratio"] = credit_reports_numerical_features["past_due_balance"].div(credit_reports_numerical_features["credit_limit"])
    credit_reports_numerical_features["current_balance_ratio"] = credit_reports_numerical_features["current_balance"].div(credit_reports_numerical_features["credit_limit"])
    return credit_reports_numerical_features.merge(credit_reports_dummy_features, how="left", on="application_id")

########## merge all features ##########
def get_features(general_info_df, internal_payments, credit_reports):
    """Get and merge all features related to applications in general_info_df dataframe"""
    # Input
    GENERAL_INFO = general_info_df.copy()

    # Get internal features
    internal_features = get_internal_features(internal_payments_df=internal_payments)
    GENERAL_INFO = GENERAL_INFO.merge(internal_features, how="left", on="application_id")

    # Get credit report features
    credit_report_features = get_credit_features(credit_reports_df=credit_reports)
    GENERAL_INFO = GENERAL_INFO.merge(credit_report_features, how="left", on="application_id")

    return GENERAL_INFO

########## Get all features ##########
features = get_features(general_info_df = clean_general_info, internal_payments = clean_internal_payments, credit_reports = clean_credit_reports)