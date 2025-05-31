# 声明逻辑斯谛回归模型
lr_model = LogisticRegression(solver='liblinear', max_iter=500)

# 假设 X_train 和 y_train 是训练数据和目标变量
# 对原始模型做交叉验证
cv_score = cross_val_score(lr_model, X_train, y_train, scoring='f1', cv=5)

# 输出交叉验证分数
print('Cross validation score of raw model: {}'.format(cv_score))
#查看下当前模型
# 查看模型信息
print(lr_model.get_params())
