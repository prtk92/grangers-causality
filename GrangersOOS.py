# -*- coding: utf-8 -*-
# """
# Created on Sat Oct 31 17:20:29 2020
# @author: prtk9
# """
import numpy as np
import pandas as pd

oos = pd.read_csv('July_2019_OOS.csv')

oos["Date"] = pd.to_datetime(oos["Date"])


oos = oos.drop_duplicates()
oos.columns


import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

# oost = oos.sort_values("Date")[oos["Machine_ID"]=="14261_4"][["Scratchers_Sales","Out_of_Stock__"]]
# gc_res = grangercausalitytests(oost, [1])
# oost["Scratchers_Sales"].corr(oost["Out_of_Stock__"])
# gc_res[1][0]['ssr_ftest'][1]
# gc_res[1][0]['ssr_chi2test'][1]
# gc_res[1][0]['lrtest'][1]
# gc_res[1][0]['params_ftest'][1]
    
machineid = oos["Machine_ID"].unique()
oosmean = pd.DataFrame(columns=['Machine_ID','Scratchers_Sales','Out_of_Stock__',
                                'Retailer_ID', 'Terminal_ID', 'termType', 'Chain_ID',
                                'TMID', 'Location_Name', 'Location_Address', 'Location_City', 'Zip',
                                'Lagged_Correlation','ssr_ftest','ssr_chi2test','lrtest','params_ftest'])
for i in machineid :
    oos_i = oos.sort_values("Date")[oos["Machine_ID"]==i][["Scratchers_Sales","Out_of_Stock__",
                                                           'Retailer_ID', 'Terminal_ID', 'termType', 'Chain_ID',
                                                           'TMID', 'Location_Name', 'Location_Address', 'Location_City', 'Zip']]
    oos_i = oos_i.reset_index()
    if((pd.isna(oos_i["Scratchers_Sales"].corr(oos_i["Out_of_Stock__"]))) or (oos_i.shape[0] < 30) or oos_i["Out_of_Stock__"].std() < 0.00000001):
        print(i,oos_i["Scratchers_Sales"].corr(oos_i["Out_of_Stock__"]),oos_i.shape[0],oos_i["Out_of_Stock__"].std())
    else:
        print(i,oos_i["Scratchers_Sales"].corr(oos_i["Out_of_Stock__"]),oos_i.shape[0],oos_i["Out_of_Stock__"].std())
        lagcorr = oos_i["Scratchers_Sales"].corr(oos_i["Out_of_Stock__"].shift(periods=1))
        gc_res = grangercausalitytests(oos_i[["Scratchers_Sales","Out_of_Stock__"]], [1])
        gc_row = pd.Series([i,
                            oos_i["Scratchers_Sales"].mean(),
                            oos_i["Out_of_Stock__"].mean(),
                            oos_i["Retailer_ID"][0],
                            oos_i["Terminal_ID"][0],
                            oos_i["termType"][0],
                            oos_i["Chain_ID"][0],
                            oos_i["TMID"][0],
                            oos_i["Location_Name"][0],
                            oos_i["Location_Address"][0],
                            oos_i["Location_City"][0],
                            oos_i["Zip"][0],
                            lagcorr,
                            gc_res[1][0]['ssr_ftest'][1],
                            gc_res[1][0]['ssr_chi2test'][1],
                            gc_res[1][0]['lrtest'][1],
                            gc_res[1][0]['params_ftest'][1]]
                           ,index = oosmean.columns)
        oosmean = oosmean.append(gc_row, ignore_index=True)

oosmean = oosmean.dropna(axis=0)
oosmean.to_csv("Grangers_2019_OOS.csv")

oos_i["Location_City"][0]

# oosg = pd.read_csv('Grangers_2019_OOS.csv')
# oosg.columns
# X = oosg[["Scratchers_Sales", "Out_of_Stock__", "termType", "Chain_ID"]]
# y = oosg["Lagged_Correlation"]

# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(categories='auto', drop=None, handle_unknown='error', sparse=False, dtype=int)
# Xenc = pd.DataFrame(ohe.fit_transform(X[["termType", "Chain_ID"]]), columns=ohe.get_feature_names())
# Xenc.head()

# Xenc = pd.concat([X, Xenc], axis=1)
# Xenc.head()

# Xenc = Xenc.drop(["termType", "Chain_ID"], axis=1)
# Xenc.head()

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(Xenc, y, test_size=0.2)

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression

# mlr = LinearRegression()

# mlr.fit(X_train, y_train)

# y_pred = mlr.predict(X_test)

# from sklearn.metrics import r2_score
# r2_score(y_test, y_pred)





# rfr = RandomForestRegressor(n_estimators=10, 
#                               max_features='auto', bootstrap=True,
#                               max_depth=None, min_samples_split=2, min_samples_leaf=1)
# rfr.fit(X_train, y_train)

# y_pred = rfr.predict(X_test)

# from sklearn.metrics import r2_score
# r2_score(y_test, y_pred)

