
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc


def label_coder(dat, ID=None):
    cols = dat.columns
    for col in cols:
        if (col not in ID) & (dat[col].dtypes=='object'):
            dat[col] = dat[col].astype('category').cat.codes
    return dat


train = pd.read_csv('application_train.csv')
test = pd.read_csv('application_test.csv')

print 'train size', len(train), 'test size', len(test)  # 307511, 48744
print 'Positive label', train['TARGET'].tolist().count(1)

df = train.append(test)
del train, test
# bureau data
bur = pd.read_csv('bureau.csv')
bur_b = pd.read_csv('bureau_balance.csv')

# bur['SK_ID_CURR'].nunique()
# bur['SK_ID_BUREAU'].nunique()
# bur_b['SK_ID_BUREAU'].nunique()
agg_funs = ['min', 'max', 'mean', 'var', 'sum']

bur_b = label_coder(bur_b, ID='SK_ID_BUREAU')
bur = label_coder(bur, ID=['SK_ID_CURR', 'SK_ID_BUREAU'])

bur_b = bur_b.groupby('SK_ID_BUREAU').agg(agg_funs).reset_index()
bur_b.columns = pd.Index(e[0]+'_'+e[1] for e in bur_b.columns)

bur = bur.merge(bur_b, how='left', left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU_')
bur.drop('SK_ID_BUREAU_', axis=1, inplace=True)
bur.drop('SK_ID_BUREAU', axis=1, inplace=True)
del bur_b
gc.collect()

bur = bur.groupby('SK_ID_CURR').agg(agg_funs).reset_index()
bur.columns = pd.Index(e[0]+'_'+e[1] for e in bur.columns)

print bur.shape
print bur.columns
bur.to_csv('bur_merge.csv', index=False)
gc.collect()

df = df.merge(bur, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR_')
df.drop('SK_ID_CURR_', axis=1 , inplace=True)

del bur
gc.collect()

ins_payment = pd.read_csv('installments_payments.csv')
ins_payment.drop('SK_ID_PREV', axis=1, inplace=True)
gc.collect()

ins_payment = label_coder(ins_payment, ID='SK_ID_PREV')
ins_payment = ins_payment.groupby('SK_ID_CURR').agg(agg_funs).reset_index()

ins_payment.columns = pd.Index(e[0]+'_'+e[1] for e in ins_payment.columns)

df = df.merge(ins_payment, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR_')
df.drop('SK_ID_CURR_', axis=1, inplace=True)
del ins_payment
gc.collect()

pc_balance = pd.read_csv('POS_CASH_balance.csv')

pc_balance.drop('SK_ID_PREV', axis=1, inplace=True)
gc.collect()

pc_balance = label_coder(pc_balance, ID='SK_ID_PREV')
pc_balance = pc_balance.groupby('SK_ID_CURR').agg(agg_funs).reset_index()

pc_balance.columns = pd.Index(e[0]+'_'+e[1] for e in pc_balance.columns)

df = df.merge(pc_balance, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR_')
df.drop('SK_ID_CURR_', axis=1, inplace=True)
del pc_balance
gc.collect()

cd_balance = pd.read_csv('credit_card_balance.csv')

cd_balance.drop('SK_ID_PREV', axis=1, inplace=True)

cd_balance = label_coder(cd_balance, ID='SK_ID_PREV')
cd_balance = cd_balance.groupby('SK_ID_CURR').agg(agg_funs).reset_index()

cd_balance.columns = pd.Index(e[0]+'_'+e[1] for e in cd_balance.columns)

df = df.merge(cd_balance, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR_')
df.drop('SK_ID_CURR_', axis=1, inplace=True)
del cd_balance
gc.collect()
df.shape

prev_app = pd.read_csv('previous_application.csv')
prev_app.drop('SK_ID_PREV', axis=1, inplace=True)
prev_app.replace(365243, np.nan, inplace=True)

prev_app = label_coder(prev_app, ID='SK_ID_PREV')
prev_app['APP_CREDIT_PERC'] = prev_app['AMT_APPLICATION']/prev_app['AMT_CREDIT']

prev_app = prev_app.groupby('SK_ID_CURR').agg(agg_funs).reset_index()

prev_app.columns = pd.Index(e[0]+'_'+e[1] for e in prev_app.columns)

df = df.merge(prev_app, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR_')
df.drop('SK_ID_CURR_', axis=1, inplace=True)
del prev_app
gc.collect()
df.shape
df.dtypes

df.to_csv('dat_all.csv', index=False)

# modeling
df = pd.read_csv('dat_all.csv')

