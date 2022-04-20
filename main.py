# Importing all the relevent libraries
import email
import glob
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              GradientBoostingClassifier, VotingClassifier, 
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
import seaborn as sns
from bs4 import BeautifulSoup as bs
import re
import warnings
warnings.filterwarnings('ignore')

# Storing the paths for all the phishing and normal emails
path = os.path.join(os.path.dirname(__file__), 'Phishing Dataset')
phish = os.path.join(path, 'phishing\\')
normal = os.path.join(path, 'normal\\')

# Selecting all the files from the phishing and normal folders
phish =glob.glob(phish + "*.*")
normal = glob.glob(normal + '*.*')


# Initialiasing the lists to store the attributes
lnks = []; dear = []; account = []; secure = []; bank = []; html = [];
verify = []; sus = []; eb_pay = []; clss = []; log = []; img = []


# Calculating and storing all the attributes in the loop below
# Initialiasing the lists to store the attributes
# Initialiasing the lists to store the attributes
lnks = []; dear = []; account = []; secure = []; bank = []; html = []; protect = []; click = [];
verify = []; sus = []; eb_pay = []; clss = []; log = []; img = []; please = []; thank = []

def process(flnme, cls):
# Calculating and storing all the attributes in the loop below
    for fl in flnme:
        # Calculating all the attribues for the phishing mails
        with open(fl, 'r', errors = 'ignore') as f:
            content = f.read()
            # Extracting the mail and its metadata from the file
            b = email.message_from_string(content)

            # Extracting the meaningful information from the message extracted
            if b.is_multipart():
                for part in b.get_payload():
                    mail = part.get_payload()
            else:
                mail = b.get_payload()

            # If I can directly access the mail then get the body else just skip the given mail
            try:
                soup = bs(mail, 'html.parser')
                body = re.sub('\s+', ' ', soup.get_text()).strip()
            except:
                continue

            # Extract all the links from the body of the mail
            imgs = soup.find_all('img')
            # links = re.findall('htt?p?s?://\S+', body)
            links = soup.find_all('a')
            htmls = soup.find_all()

            # Convert all the strings in the body to lower case
            body = body.lower()
            
            # Store the number of links in the lnks variable
            html.append(len(htmls))
            lnks.append(len(links))
            img.append(len(imgs))
            
            # Check if the mail contains the word dear in it
            if 'dear' in body:
                dear.append(1)
            else:
                dear.append(0)
            
            # Check if the word login is in the body
            if 'login' in body or 'log in' in body or 'log-in' in body:
                log.append(1)
            else:
                log.append(0)  

            # Check if the mail contains the word account in it
            if 'account' in body:
                account.append(1)
            else:
                account.append(0)


            # Check if the mail contains the word bank in it
            if 'bank' in body:
                bank.append(1)
            else:
                bank.append(0)  
                
            # Check if the mail contains the word please in it
            if 'please' in body:
                please.append(1)
            else:
                please.append(0) 
            
            # Check if the mail contains the word protect in it
            if 'protect' in body:
                protect.append(1)
            else:
                protect.append(0) 
            
            # Check if the mail contains the word click in it
            if 'click' in body or 'select' in body or 'visit' in body:
                click.append(1)
            else:
                click.append(0)

            # Check if the mail contains the word thank you in it
            if 'thank you' in body:
                thank.append(1)
            else:
                thank.append(0)    
            
            # Check if the mail contains the word secure or security in it        
            if 'secure' in body or 'security' in body:
                secure.append(1)
            else:
                secure.append(0)
                
            # Check if the mail contains the word suspend or suspended in it
            if 'suspend' in body or 'suspended' in body:
                sus.append(1)
            else:
                sus.append(0)
                
            # Check if the mail contains the word verify in it
            if 'verify' in body:
                verify.append(1)
            else:
                verify.append(0)
            
            # Check if the mail contains the word ebay or paypal in it
            if 'ebay' in body or 'paypal' in body:
                eb_pay.append(1)
            else:
                eb_pay.append(0)
                

            # Class for the phishing mails is 1
            clss.append(cls)
        

process(phish, 1)  # for phishing mails
process(normal, 0) # for normal mails

# Convert all the data calculated and extracted above in a dictionary
data = {'Links':lnks, 
        'Dears': dear, 
        'Account': account, 
        'HTML': html, 
        # 'Secure': secure, 
        # 'Login': log, 
        'Protect': protect, 
        # 'Verify': verify, 
        'Click': click,
        # 'Suspend': sus, 
        'Images': img, 
        'Bank': bank, 
        'Please': please, 
        'Thank You': thank, 
        'Ebay and Paypal': eb_pay, 
        'Target': clss}
df = pd.DataFrame(data)

# Convery the dictionary into a dataframe
df = pd.DataFrame(data)


# Create the X and y variables for training 
X = df.drop(['Target'], axis = 1)
y = df['Target']

# Split the data into training and testing set with 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state = 42, stratify = y)


# Scale the links attribute as it is numerical
ss = StandardScaler()
columns = ['Links', 'Images', 'HTML']
X_train.loc[:, columns] = ss.fit_transform(X_train[columns])
X_test.loc[:, columns] = ss.transform(X_test[columns])



# Apply grid search to find the best hyperparameter for support vector classifier
param = [
    {
        'kernel': ['rbf'], 
        'C': [0.1, 0.3, 1, 2, 3, 4], 
        'gamma': [0.3, 1, 3, 10, 12, 15, 25, 28]
    }, 
]

svc = SVC(probability = True)
gs_svc = GridSearchCV(svc, param, cv = 5, n_jobs = -1, verbose = 1)
gs_svc.fit(X_train, y_train)
svc_best = gs_svc.best_estimator_

# Store the accuracy of the best estimator
svc_acc = svc_best.score(X_test, y_test)


# Apply grid search to find the best hyperparameter for the xboost classifier
param_grid={
    'max_depth': range(2, 10, 2),
    'n_estimators': range(26, 46, 2),
    'learning_rate': [0.2, 0.1, 0.03, 0.01]
}

xg = XGBClassifier(eval_metric='logloss', n_jobs = -1, tree_method = 'gpu_hist', use_label_encoder = False)
gs_xg = GridSearchCV(xg, param_grid, cv = 5, n_jobs = -1, verbose = 1)
gs_xg.fit(X_train, y_train)

xg_best = gs_xg.best_estimator_
xg_acc =  xg_best.score(X_test, y_test)


# Apply grid search to find the best hyperparameter for the gradient boost classifier
param = [
    {'n_estimators': range(12, 36, 4), 
     'max_depth': range(8, 24, 4),
     'max_leaf_nodes': range(8, 28, 4),
    }
]

gb = GradientBoostingClassifier()
gs_gb = GridSearchCV(gb, param, cv = 5, n_jobs = -1, verbose = 1)
gs_gb.fit(X_train, y_train)

gb_best = gs_gb.best_estimator_
gb_acc = gb_best.score(X_test, y_test)


# Apply grid search to find the best hyperparameter for the ranfom forest classifier
param = [
    {'n_estimators': [100, 200, 300, 400, 450, 500], 
     'max_depth': [3, 4, 6, 8, 10, 12], 
     'max_leaf_nodes': [15, 20, 25]}, 
]

rf = RandomForestClassifier()
gs_rf = GridSearchCV(rf, param, cv = 5, n_jobs = -1, verbose = 1)
gs_rf.fit(X_train, y_train)

rf_best = gs_rf.best_estimator_
rf_acc = rf_best.score(X_test, y_test)


# Apply grid search to find the best hyperparameter for the Decision Tree classifier
param = [
    {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(4, 20, 2),
        'max_leaf_nodes': range(20, 44, 2),
    }
]

dt = DecisionTreeClassifier()
gs_dt = GridSearchCV(dt, param, cv = 5, n_jobs = -1, verbose = 1)
gs_dt.fit(X_train, y_train)

dt_best = gs_dt.best_estimator_
dt_acc = dt_best.score(X_test, y_test)



# Apply grid search to find the best hyperparameter for the adaboost classifier
param = [
    {'n_estimators': [50, 100, 150, 200, 300, 400]}
]
ada = AdaBoostClassifier()
gs_ada = GridSearchCV(ada, param, cv = 5, n_jobs = -1, verbose = 1)
gs_ada.fit(X_train, y_train)

ada_best = gs_ada.best_estimator_
ada_acc = ada_best.score(X_test, y_test)



# Apply grid search to find the best hyperparameter for the extra trees classifier
param = [
    {'n_estimators': range(8, 28, 4), 
     'max_depth': range(24, 48, 4),
     'max_leaf_nodes': range(20, 48, 4),
    }
]

et = ExtraTreesClassifier()
gs_et = GridSearchCV(et, param, cv = 5, n_jobs = -1, verbose = 1)
gs_et.fit(X_train, y_train)

et_best = gs_et.best_estimator_
et_acc = et_best.score(X_test, y_test)

# Calculating the feature importance for all the attributes in the dataset I created
fi = pd.Series(et_best.feature_importances_, index = X_train.columns)



# Apply grid search to find the best hyperparameter for the knn classifier
param = [
    {'n_neighbors': range(2, 14, 1)}
]

knn = KNeighborsClassifier()
gs_knn = GridSearchCV(knn, param, cv = 5, n_jobs = -1)
gs_knn.fit(X_train, y_train)

knn_best = gs_knn.best_estimator_
knn_acc = knn_best.score(X_test, y_test)



# Apply grid search to find the best hyperparameter for the voting classifier
vc_lr = LogisticRegression(solver = 'sag')
vc_mlp = MLPClassifier()

vc = VotingClassifier(estimators = [('rf', rf_best), ('svc', svc_best), ('lr', vc_lr), 
                                    ('mlp', vc_mlp), ('xgc', xg_best), ('knn', knn_best), 
                                    ('ET', et_best), ('dt', dt_best),
                                    ('gb', gb_best)], 
                                    voting = 'soft', n_jobs = -1, verbose = 1)
vc.fit(X_train, y_train), vc.score(X_test, y_test)
vc_acc = vc.score(X_test, y_test)


models = ['SVC', 'RandomForest', 'XGBoost', 'Gradient Boost', 'AdaBoost', 'Decision Tree', 'Extra Trees', 'KNN', 'Voting Classifier']
acc = [svc_acc, rf_acc, xg_acc, gb_acc, ada_acc, dt_acc, et_acc, knn_acc, vc_acc]

# Printing the accuracies of all the individual models
print('Accuracies: ')
for i in range(len(models)):
    print(models[i], '-', acc[i])

pred = gs_et.predict(X_test)

# Calculating and plotting the confusion matrix for voting classifier predictions
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot = True, fmt = 'g');
plt.show()

# Printing the classification report
print(classification_report(y_test, pred))


print('Important Features:')
print(fi)

p = {}
for i in range(len(models)):
    p[models[i]] = acc[i]

# Plotting the accuracy comparison chart
p = dict(sorted(p.items(), key = lambda x: x[1], reverse = True))
plt.figure(figsize = (8, 6))
plt.xlabel('Accuracy')
sns.barplot(x = list(p.values()), y = list(p.keys()), orient = 'h');
plt.show()

