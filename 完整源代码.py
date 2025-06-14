# 警告处理 
import warnings
warnings.filterwarnings('ignore')
# 在Jupyter上画图
%matplotlib inline
#引入相关支持的包
import pandas as pd
import numpy as np
#可以事先将提供的数据集存储在本地，此处对该地址进行引用即可
#注意文件所存放路径不要有中文，容易报错
df= pd.read_csv(r'pima-indians-diabetes.data',sep=',')
#并观察熟悉数据
print(df.shape)
df.head()
df.info()
df['Outcome'].value_counts()
#导入相关包
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
#计算特征相关性并可视化
corr_matrix = df.corr(method='spearman') # pearson 皮尔逊, 是spearman 斯皮尔曼 
plt.figure(figsize=(25, 15))
sns.heatmap(corr_matrix, annot= True)
plt.show()
import math
# 绘制每个特征的分布
def plot_distribution(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    print("Available styles:", plt.style.available)
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace) # 调整图表位置和大小间距
    rows = math.ceil(float(dataset.shape[1]) / cols)# ceil方法向上取整
    for i, column in enumerate(dataset.columns): #返回索引和列名
        ax = fig.add_subplot(rows, cols, i + 1)# 创建子图，类似于subplot方法
        ax.set_title(column)	# 设置轴的标题
        if dataset.dtypes[column] == object: # 通过列的类型来区分所选取的图像类型
            g = sns.countplot(y=column, data=dataset)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25) 
        else:
            g = sns.distplot(dataset[column])
            plt.xticks(rotation=25)
plot_distribution(df, cols=3, width=20, height=20, hspace=0.45, wspace=0.5)
plt.show()
from sklearn.model_selection import train_test_split
from collections import Counter
x_cols = [col for col in df.columns if col!='Outcome']
y_col = 'Outcome'
X=df[x_cols].values    #dataframe 转化为 ndarray,才能进入下面的标准化和3D制图
y=df[y_col].values
### 绘制3D散点图-3个维度的数据点的散点分布
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for c,m,i,l in zip('rg','sx',np.unique(y),['class_1','class_2']):
    ax.scatter(X[y==i ,0], X[y==i, 4], X[y==i, 6],c=c,marker=m, label=l)
ax.set_xlabel(df.columns[1])
ax.set_ylabel(df.columns[5])
ax.set_zlabel(df.columns[7])
ax.set_title("pima-indians-diabetes")
plt.legend()
plt.show()
### 对输入特征进行降维处理
from sklearn.decomposition import PCA
from sklearn import preprocessing                    #调用标准化模块
X_std = preprocessing.scale(X)                        #降维训练前需要对数据标准化
pca = PCA(n_components=0.99, random_state=50 ) # 保留99%信息的主成分个主成分
X_pca =pca.fit(X_std).transform(X_std)
print('the Top 99% variance_ratio:',pca.explained_variance_ratio_)
X_pca.shape
### 输出降维后的前3个主成分的分布图
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
for  c, m ,i,l in zip('rg','sx',np.unique(y),['class_1','class_2']):
    ax.scatter(X_pca[y==i, 0], X_pca[y==i,1],X_pca[y==i,2], c=c, label=l, marker=m) # 散点图
ax.set_xlabel('X_pca1')
ax.set_ylabel('X_pca2')
ax.set_zlabel('X_pca3')
ax.set_title("PCA")
plt.legend(loc='lower left')
plt.show()
from collections import Counter
from sklearn.model_selection import train_test_split
### 选择不过滤和降维的全部数据进行训练
X_train, X_test, y_train, y_test = train_test_split(
                    df[x_cols],
                    df[y_col],
                    test_size=0.1,                #分割比例
                    random_state=42,              #随机数种子
                    shuffle=True,                 #是否打乱顺序 
                   stratify=df[y_col]                #指定以Target的比例做分层抽样
)               
print('Distribution of y_train {}'.format(Counter(y_train)))
print('Distribution of y_test {}'.format(Counter(y_test)))
#引入逻辑斯谛回归和交叉验证的库
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#引入评价指标的库
from sklearn.metrics import f1_score
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
c_range=[0.001,0.01,0.1,1.0]
solvers = ['liblinear','lbfgs','newton-cg','sag']
max_iters=[80,100,150,200,300]
tuned_parameters= dict(solver=solvers, C=c_range,max_iter=max_iters)
#网格搜素
from sklearn.model_selection import GridSearchCV
grid= GridSearchCV(lr_model, tuned_parameters, cv=5, scoring='f1')
grid.fit(X_train,y_train)
print('best score for model {}'.format(grid.best_score_))
print('best parameters for model {}'.format(grid.best_params_))
print('best parameters for model {}'.format(grid.best_estimator_))
#### 根据选择后的参数，最后预测
lr_model_final = LogisticRegression(C=1.0, max_iter=80, solver='newton-cg')
lr_model_final.fit(X_train,y_train)
y_train_pred = lr_model_final.predict(X_train)
print('final score of model version2: {}'.format(f1_score(y_train,y_train_pred)))
