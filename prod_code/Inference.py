# Import libraries
import pickle
from sklearn.metrics import confusion_matrix, make_scorer
from Featurizing import features

# Define a function to calculate specificity
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity

# Make a scorer from the custom specificity function
specificity_scorer = make_scorer(specificity, greater_is_better=True)

# Define colums order
cols=['num_prev_contracts', 'avg_notional', 'pct_late_payments',
       'internal_credit_payments', 'open_accounts', 'closed_accounts',
       'max_credit_amount', 'current_balance', 'past_due_balance',
       'total_credit_payments', 'worst_delinquency_past_due_balance',
       'credit_limit', 'past_due_ratio', 'current_balance_ratio',
       'institution_ARRENDADORA', 'institution_AUTOFINANCIAMIENTO',
       'institution_AUTOMOTRIZ', 'institution_BANCO',
       'institution_CAJAS DE AHORRO', 'institution_CASA DE EMPENO',
       'institution_CIA Q  OTORGA', 'institution_COBRANZA',
       'institution_COMERCIAL',
       'institution_COMPANIA DE FINANCIAMIENTO AUTOMOTRIZ',
       'institution_COMPANIA DE PRESTAMO PERSONAL',
       'institution_COMUNICACIONES', 'institution_COOPERATIVA',
       'institution_COOPERATIVA DE AHORRO Y CREDITO', 'institution_EDUCACION',
       'institution_FACTORAJE', 'institution_FINANCIERA',
       'institution_FONDOS Y FIDEICOMISOS', 'institution_GOBIERNO',
       'institution_HIPOTECAGOBIERNO', 'institution_HIPOTECARIA',
       'institution_HIPOTECARIO NO BANCARIO', 'institution_KONFIO',
       'institution_MERCANCIA PARA HOGAR Y OFICINA',
       'institution_MERCANCIA PARA LA CONSTRUCCION',
       'institution_MIC CREDITO PERS', 'institution_OTRAS FINANCIERA',
       'institution_SERVICIO DE TELEVISION DE PAGA', 'institution_SERVICIOS',
       'institution_SERVS. GRALES.',
       'institution_SOCIEDAD FINANCIERA DE OBJETO MULTIPLE',
       'institution_SOCIEDADES FINANCIERAS POPULARES',
       'institution_SOFOL AUTOMOTRIZ', 'institution_SOFOL EMPRESARIAL',
       'institution_SOFOL HIPOTECARIA', 'institution_SOFOL PRESTAMO PERSONAL',
       'institution_TELEFONIA', 'institution_TIENDA',
       'institution_UNION DE CREDITO', 'institution_VENTA POR CATALOGO',
       'institution_VENTA POR CORREO / TELEFONO',
       'account_type_Crédito Refaccionario',
       'account_type_Crédito de Habilitación de Avío', 'account_type_Hipoteca',
       'account_type_Pagos Fijos', 'account_type_Quirografiario',
       'account_type_Revolvente', 'account_type_Sin Límite Preestablecido',
       'credit_type_Arrendamiento', 'credit_type_Arrendamiento Automotriz',
       'credit_type_Banca Comunal', 'credit_type_Bienes Raíces',
       'credit_type_Compra de Automóvil', 'credit_type_Consolidación',
       'credit_type_Crédito Fiscal', 'credit_type_Crédito Personal al Consumo',
       'credit_type_Crédito al Consumo', 'credit_type_Desconocido',
       'credit_type_Factoraje', 'credit_type_Fianza',
       'credit_type_Física Actividad Empresarial',
       'credit_type_Grupo Solidario', 'credit_type_Hipotecario O Vivienda',
       'credit_type_Línea de Crédito',
       'credit_type_Línea de Crédito Reinstalable',
       'credit_type_Mejoras a la Casa',
       'credit_type_Otros (Múltiples Créditos)',
       'credit_type_Préstamo de Nomina', 'credit_type_Préstamo Empresarial',
       'credit_type_Préstamo Garantizado', 'credit_type_Préstamo Personal ',
       'credit_type_Préstamo Quirografiario',
       'credit_type_Préstamo no garantizado',
       'credit_type_Préstamo para estudiante',
       'credit_type_Tarjeta Departamental', 'credit_type_Tarjeta de Crédito']

# Open the pickle file and load the model
with open('classification_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Define the prediction function
def get_prediction(features_df):
    """Get prediction from features_dataframe"""
    FEATURES = features_df.copy()
    X = FEATURES[cols]

    # Convert bool to category
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype('category')

    Scores = rf_model.predict_proba(X)[:,0]
    return Scores

########## Get scores ##########
scores = get_prediction(features_df=features)
