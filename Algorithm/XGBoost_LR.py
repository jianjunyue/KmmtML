# # coding=UTF-8
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# import warnings
# warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
# from sklearn.metrics import make_scorer
# from xgboost.sklearn import XGBClassifier
# from sklearn import metrics
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
#
#
# def lgb_lr(data):
#     '''Descr:输入:已经构建好特征的数据data
#                 输出:xgb,xgb+lr模型的AUC比较，logloss比较
#
#
#         '''
#     train = data[(data['day'] >= 18) & (data['day'] <= 23)]
#     test = data[(data['day'] == 24)]
#     drop_name = ['is_trade',
#                  'item_category_list', 'item_property_list',
#                  'predict_category_property',
#                  'realtime'
#                  ]
#     col = [c for c in train if
#            c not in drop_name]
#     X_train = train[col]
#     y_train = train['is_trade'].values
#     X_test = test[col]
#     y_test = test['is_trade'].values
#     xgboost =XGBClassifier(n_estimators=300, max_depth=4, seed=5,
#                                 learning_rate=0.11, subsample=0.8,
#                                 min_child_weight=6, colsample_bytree=.8,
#                                 scale_pos_weight=1.6, gamma=10,
#                                 reg_alpha=8, reg_lambda=1.3, silent=False,
#                                 eval_metric='logloss')
#     xgboost.fit(X_train, y_train)
#     y_pred_test = xgboost.predict_proba(X_test)[:, 1]
#     xgb_test_auc = roc_auc_score(y_test, y_pred_test)
#     print('xgboost test auc: %.5f' % xgb_test_auc)
#     # y_tes = test['is_trade'].values
#     # xgboost编码原有特征
#     X_train_leaves = xgboost.apply(X_train)
#     X_test_leaves = xgboost.apply(X_test)
#
#     # 合并编码后的训练数据和测试数据
#     All_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
#     All_leaves = All_leaves.astype(np.int32)
#
#     # 对所有特征进行ont-hot编码
#     xgbenc = OneHotEncoder()
#     X_trans = xgbenc.fit_transform(All_leaves)
#
#     (train_rows, cols) = X_train_leaves.shape
#
#     # 定义LR模型
#     lr = LogisticRegression()
#     # lr对xgboost特征编码后的样本模型训练
#     lr.fit(X_trans[:train_rows, :], y_train)
#     # 预测及AUC评测
#     y_pred_xgblr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
#     xgb_lr_auc1 = roc_auc_score(y_test, y_pred_xgblr1)
#     print('基于Xgb特征编码后的LR AUC: %.5f' % xgb_lr_auc1)
#
#     # 定义LR模型
#     lr = LogisticRegression(n_jobs=-1)
#     # 组合特征
#     X_train_ext = hstack([X_trans[:train_rows, :], X_train])
#     X_test_ext = hstack([X_trans[train_rows:, :], X_test])
#
#     # lr对组合特征的样本模型训练
#     lr.fit(X_train_ext, y_train)
#
#     # 预测及AUC评测
#     y_pred_xgblr2 = lr.predict_proba(X_test_ext)[:, 1]
#     xgb_lr_auc2 = roc_auc_score(y_test, y_pred_xgblr2)
#     print('基于组合特征的LR AUC: %.5f' % xgb_lr_auc2)
#
#     # -------------------计算logloss
#     pred = pd.DataFrame()
#     pred['is_trade'] = y_test
#     pred['xgb_pred'] = y_pred_test
#     pred['xgb_lr_pred'] = y_pred_xgblr1
#     logloss1 = log_loss(pred['is_trade'], pred['xgb_pred'])
#     logloss2 = log_loss(pred['is_trade'], pred['xgb_lr_pred'])
#     print('xgb   logloss:' + str(logloss1))
#     print('xgb+lr   logloss:' + str(logloss2))