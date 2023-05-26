# -*- coding: utf-8 -*-

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

root_path = 'datos/'
income_statement = pd.read_csv(root_path + 'income_statement_inliers.csv', index_col=0)
balance_sheet = pd.read_csv(root_path + 'balance_sheet_inliers.csv', index_col=0)
cash_flow = pd.read_csv(root_path + 'cash_flow_inliers.csv', index_col=0)
other_characteristics = pd.read_csv(root_path + 'other_characteristics_inliers.csv', index_col=0)
quarterly_performance = pd.read_csv(root_path + 'quarterly_performance_inliers.csv', index_col=0)
ratios = pd.read_csv(root_path + 'ratios_without_infinites.csv', index_col=0)
quarterly_performance_ratios = pd.read_csv(root_path + 'quarterly_performance_for_ratios.csv', index_col=0)

income_statement_numerical = income_statement.iloc[:,2:]
balance_sheet_numerical = balance_sheet.iloc[:,2:]
cash_flow_numerical = cash_flow.iloc[:,2:]
other_characteristics_numerical = other_characteristics.iloc[:,2:]
ratios_numerical = ratios.iloc[:,2:]
quarterly_performance_perf = quarterly_performance['Performance']
quarterly_performance_ratios_perf = quarterly_performance_ratios['Performance']

income_statement_numerical = income_statement_numerical.loc[ratios_numerical.index]
balance_sheet_numerical = balance_sheet_numerical.loc[ratios_numerical.index]
cash_flow_numerical = cash_flow_numerical.loc[ratios_numerical.index]
other_characteristics_numerical = other_characteristics_numerical.loc[ratios_numerical.index]
quarterly_performance_perf = quarterly_performance_perf.loc[ratios_numerical.index]

income_statement_numerical.reset_index(inplace=True, drop=True)
balance_sheet_numerical.reset_index(inplace=True, drop=True)
cash_flow_numerical.reset_index(inplace=True, drop=True)
other_characteristics_numerical.reset_index(inplace=True, drop=True)
ratios_numerical.reset_index(inplace=True, drop=True)

quarterly_performance_perf.reset_index(inplace=True, drop=True)
quarterly_performance_ratios_perf.reset_index(inplace=True, drop=True)

"""## Deleting values

### Data far away from distribution
"""

stock_performance = quarterly_performance_perf[quarterly_performance_perf <= 1]
stock_performance_for_ratios = quarterly_performance_ratios_perf[quarterly_performance_ratios_perf <= 1]

income_statement_numerical_2 = income_statement_numerical.loc[stock_performance.index]
balance_sheet_numerical_2 = balance_sheet_numerical.loc[stock_performance.index]
cash_flow_numerical_2 = cash_flow_numerical.loc[stock_performance.index]
other_characteristics_numerical_2 = other_characteristics_numerical.loc[stock_performance.index]
ratios_numerical_2 = ratios_numerical.loc[stock_performance_for_ratios.index]

other_characteristics_numerical_2 = other_characteristics_numerical_2.iloc[:,[0,5,6,7]]


income_statement_numerical_2 = income_statement_numerical_2.drop(columns=['incomeTaxExpense'])
balance_sheet_numerical_2 = balance_sheet_numerical_2.drop(columns=['longTermDebt', 'currentLongTermDebt'])
cash_flow_numerical_2 = cash_flow_numerical_2.drop(columns=['changeInInventory', 'dividendPayout', 'dividendPayoutCommonStock'])

"""### Columns from feature importance"""

income_statement_numerical_2 = income_statement_numerical_2.loc[:,['grossProfit', 'totalRevenue', 'operatingIncome', 'netInterestIncome',
       'interestIncome', 'interestExpense', 'incomeBeforeTax',
       'interestAndDebtExpense', 'comprehensiveIncomeNetOfTax', 'ebit', 'ebitda', 'netIncome']]

balance_sheet_numerical_2 = balance_sheet_numerical_2.loc[:,['totalAssets', 'totalCurrentAssets', 'currentNetReceivables',
       'totalNonCurrentAssets', 'propertyPlantEquipment',
       'longTermInvestments', 'otherCurrentAssets', 'otherNonCurrentAssets',
       'totalCurrentLiabilities', 'currentAccountsPayable',
       'otherCurrentLiabilities', 'otherNonCurrentLiabilities',
       'totalShareholderEquity', 'retainedEarnings', 'commonStock']]

cash_flow_numerical_2 = cash_flow_numerical_2.loc[:,['operatingCashflow', 'paymentsForOperatingActivities',
       'changeInOperatingAssets', 'depreciationDepletionAndAmortization',
       'capitalExpenditures', 'changeInReceivables', 'profitLoss',
       'cashflowFromInvestment', 'changeInCashAndCashEquivalents',
       'netIncome']]

ratios_numerical_2 = ratios_numerical_2.loc[:,['priceToEarningsRatio', 'priceToSalesRatio', 'priceToBookatio',
       'debtToEquityRatio', 'returnOnAssets', 'earningsPerShare']]

"""## Joining tables"""

data = pd.concat([income_statement_numerical_2, balance_sheet_numerical_2,
                  cash_flow_numerical_2.drop(columns=['netIncome']), other_characteristics_numerical_2,
                  ratios_numerical_2],
                 axis=1)

data = data.drop(columns=['reportedEPS'])

data_2 = data[['operatingIncome', 'incomeBeforeTax', 'ebit', 'ebitda', 'netIncome',
       'operatingCashflow', 'profitLoss', 'open', 'priceToEarningsRatio',
       'earningsPerShare']]

data_2 = data_2.drop(columns=['profitLoss'])

"""## Scaling data (minMax)"""

minMax_scaler = MinMaxScaler()
minMax_scaler.fit(data_2)
data_minMax = minMax_scaler.transform(data_2)
data_minMax = pd.DataFrame(data_minMax, columns= data_2.columns)

data_minMax = data_minMax.replace(0, 1e-8)

data_minMax.min()

minMax_scaler_output = MinMaxScaler()
minMax_scaler_output.fit(stock_performance_for_ratios.values.reshape(-1, 1))
stock_performance_ratios_minMax = minMax_scaler_output.transform(stock_performance_for_ratios.values.reshape(-1, 1))

stock_performance_ratios_minMax[stock_performance_ratios_minMax == 0] = 1e-8

"""## Data transformation"""

data_2_log = np.log(data_minMax)

stock_performance_2_sqrt = np.sqrt(stock_performance_ratios_minMax)

"""# Model with RFR
"""

media = data_2_log.mean()
desviacion_estandar = data_2_log.std()

# Calcular los límites superiores e inferiores
limite_superior = media + 2 * desviacion_estandar
limite_inferior = media - 2 * desviacion_estandar

# Identificar las filas que superan los límites en cada columna
filas = ((data_2_log > limite_superior) | (data_2_log < limite_inferior)).any(axis=1)

outliers_2 = np.array(filas)
non_outliers_2 = [not valor for valor in outliers_2]

data_3_log = data_2_log[non_outliers_2]
stock_performance_3_sqrt = stock_performance_2_sqrt[non_outliers_2]

"""## Creating the model"""

x_train_d, x_test_d, y_train_d, y_test_d = train_test_split(data_3_log, stock_performance_3_sqrt,
                                                              test_size = 0.2,
                                                              random_state = 15)

y_train_d, y_test_d = y_train_d.ravel(), y_test_d.ravel()

"""{'max_depth': 10, 'max_features': 5, 'n_estimators': 500}"""

rfr_d = RandomForestRegressor(max_depth=10, max_features=5, n_estimators=500)
rfr_d.fit(x_train_d, y_train_d)

"""### Comparing output after scaling"""

y_min = stock_performance_3_sqrt.min()
y_max = stock_performance_3_sqrt.max()

# y_min = y_test_d.min()
# y_max = y_test_d.max()

def scale_input(X):
       X_minMax = minMax_scaler.transform(X)
       X_minMax = X_minMax.replace(0, 1e-8)
       X_log = np.log(X_minMax)
       return X_log


def scale_output(y_pred):
       minMax_scaler_pred = MinMaxScaler(feature_range=(y_min, y_max))
       minMax_scaler_pred.fit(y_pred.reshape(-1, 1))
       y_pred_scaled = minMax_scaler_pred.transform(y_pred.reshape(-1, 1))
       #Revert square root transformation
       y_pred_scaled_reverted_sqrt = y_pred_scaled.apply(lambda x: x ** 2)
       #Revert minMax transformation
       y_pred_final = minMax_scaler_output.inverse_transform(y_pred_scaled_reverted_sqrt)
       return y_pred_final

pipeline = Pipeline([
    ('input_scaler', FunctionTransformer(scale_input)),
    ('regression_model', rfr_d),
    ('output_scaler', FunctionTransformer(scale_output))
])

joblib.dump(pipeline, 'modelo/modelo.pkl')