# test the usage of torch dp (opacus) according to NDSS paper.
import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, log_loss, accuracy_score
# below are required for DP
from torch.utils.data import Dataset, DataLoader
from opacus import PrivacyEngine
import opacus
import shap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FintrustML(nn.Module):  # -wh general machine learning model for FinTrust
    def __init__(self, dims, outdims):
        super(FintrustML, self).__init__()
        # attributes need to be added here? -wh
        self.__name__ = "finmodel"

        self.finmodel = nn.Sequential( # note the name is finmodel
            nn.Linear(dims, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, outdims),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.finmodel(x)
        return output

class FLDataset(Dataset): # inherited from the parent abstract class Dataset
    def __init__(self, pdframe_to_convert, target_dim):
        pdframe = pdframe_to_convert # The input is a pandas dataframe
        x = pdframe.iloc[:, :-target_dim].values # all the features
        y = pdframe.iloc[:, -target_dim:].values # all the targets
        self.x_dataset = torch.tensor(x, dtype=torch.float)
        self.y_dataset = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.y_dataset)
    
    def __getitem__(self, idx):
        return self.x_dataset[idx], self.y_dataset[idx]

softmax = torch.nn.Softmax(dim=1)
# exp parameters
batch_size = 64
rounds = 500
############################ Differential Privacy ####################################
Diff_Privacy = False # !enable or disable DP
EPSILON = 1.0 # !privacy budget

DELTA = 1e-5 # The target δ of the (ϵ,δ)-differential privacy guarantee. Generally, it should be set to be less than the inverse of the size of the training dataset. 
MAX_GRAD_NORM = 1.0 # The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step.
# record logs
selected_opt = 'adam' # ! optimizer function and learning rate
selected_lr = 1e-3
str_lr = '{:.0e}'.format(selected_lr) # otherwise 1e-3 turns to '0.001'
log_name = f'{selected_opt}_{str_lr}/epsilon{EPSILON}' # ! dir name of the logs (for TensorBoard)
print(log_name)
if Diff_Privacy == True:
    log_path = f'./logs/log_torchdp/dp_enabled/{log_name}'
else:
    log_path = f'./logs/log_torchdp/dp_disabled/{log_name}'
os.makedirs(os.path.dirname(log_path), exist_ok=True) # create folder if not exist
writer = SummaryWriter(log_path)

target = 'personal_loan' # target column
selected_variables = ['age','experience','income','family','ccavg','education','mortgage','securities_account','cd_account','online','creditcard','personal_loan']
categorical_variables = ['family','education','securities_account','cd_account','online','creditcard','personal_loan']
categorical_variables.append(categorical_variables.pop(categorical_variables.index(target))) # move the target column to the end
print(categorical_variables)
continuous_variables = [col for col in selected_variables if col not in categorical_variables]
print(continuous_variables)

orig_data = pd.read_csv('./sourceData/bank_Loan.csv', usecols=selected_variables)
# test_cols = list(pd.read_csv('./sourceData/bank_Loan.csv')[selected_variables])
# print('test_cols: \n', test_cols)
target_num = orig_data[target].nunique() # possible classes of target

print('Output dimension: ', target_num)
print('Classes to predict:\n', orig_data[target].value_counts())
# preprocessing, onehot, minmaxscale
encoded_data = pd.get_dummies(orig_data, prefix=categorical_variables, columns=categorical_variables, prefix_sep='_')
scaler = MinMaxScaler(feature_range=(-1, 1))
encoded_data[continuous_variables] = scaler.fit_transform(encoded_data[continuous_variables])

print('encoded data head: \n', encoded_data.head())
print('encoded data type: ', type(encoded_data))
print('encoded data shape: ', encoded_data.shape)
trainSet = encoded_data.sample(frac=0.7, random_state=16)
testSet = encoded_data.drop(trainSet.index)

# Use PyTorch DataLoader to load dataset
column_names = trainSet.columns.values # a list of strings
print('Column names:\n', len(column_names), column_names) 
train_dataset = FLDataset(pdframe_to_convert=trainSet, target_dim=target_num)
print('field_names of train_dataset: \n', train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = FLDataset(pdframe_to_convert=testSet, target_dim=target_num)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# just for testing the impact of data_loader on calcuation of epsilon
# encoded_data = encoded_data.sample(frac=0.5, random_state=12)
# all_dataset = FLDataset(pdframe_to_convert=encoded_data, target_dim=target_num)
# all_loader = DataLoader(all_dataset, batch_size=batch_size)

train_X, train_y = next(iter(train_loader)) # create a iterator and get the items 1by1
train_X = train_X.to(device)
train_y = train_y.to(device)
print(train_X.shape[1], train_y.shape[1])

total_train_step = 0
total_train_loss = 0
total_test_step = 0
total_test_loss = 0
# hyperparameters of finmodel
finmodel = FintrustML(dims=train_X.shape[1], outdims=train_y.shape[1]).to(device)
if target_num == 2:
    loss_fn = nn.BCELoss() # for binary classification
else:
    loss_fn = nn.CrossEntropyLoss() # for multi-classification

if selected_opt == 'adam':
    optimizer = torch.optim.Adam(finmodel.parameters(), lr=selected_lr)
elif selected_opt == 'sgd':
    optimizer = torch.optim.SGD(finmodel.parameters(), lr=selected_lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # dynamic lr

model_weights = finmodel.state_dict()
print('FinModel before DP:', finmodel)
# assuming that we know the privacy budget (epsilon)
if Diff_Privacy == True:
    privacy_engine = PrivacyEngine()
    finmodel, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=finmodel,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=rounds,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

# finmodel = finmodel.to_standard_module()
# print('FinModel unwrapped:', finmodel)

# finmodel.load_state_dict(model_weights)
# start training:
for i in range(rounds):
    print("-----------start training round {}------------".format(i+1))
    finmodel.train()
    total_train_loss = []
    # for observing the architecture of neural network
    # for name, param in finmodel.named_parameters():
    #     print(name, param.data)
    # print('length of finmodel.parameters() list', len(list(finmodel.parameters())))
    # for i in range(len(list(finmodel.parameters()))):
    #     print(f'size of {i}-th item:', list(finmodel.parameters())[i].size())
    #     print(list(finmodel.parameters())[i])
    for j, (features, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        features = features.to(device)
        targets = targets.to(device)
        outputs = finmodel(features) # [batch_size, train_y.shape[1]]
        # print('output', outputs[0])
        # targets = torch.argmax(targets, dim=1) # 1D array
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss.append(loss.item())
    avg_train_loss = np.average(total_train_loss)
    total_train_step += 1

    # get epsilon if Differential Privacy is enabled
    if Diff_Privacy == True:
        epsilon = privacy_engine.get_epsilon(DELTA)
        print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")
        print(f"Train step {total_train_step}: ϵ={epsilon:.3f},  δ={DELTA}")
        writer.add_scalar("epsilon", epsilon, total_test_step)
    # record train loss
    print("Loss of train step {}: {}".format(total_train_step, avg_train_loss))
    writer.add_scalar("avg train loss", avg_train_loss, total_train_step)
    # scheduler.step()
    # start testing:
    finmodel.eval()
    with torch.no_grad():
        pred_targets = []
        test_targets = []
        pred_y_auc = []
        for k, (t_features, t_targets) in enumerate(test_loader):
            t_features = t_features.to(device)
            t_targets = t_targets.to(device)
            t_targets = torch.argmax(t_targets, dim=1)
            output = finmodel(t_features)
            pred_y_auc.append(output)
            mask = torch.argmax(output, dim=1)
            pred_targets.extend(mask)
            test_targets.extend(t_targets)
        
        pred_targets = torch.tensor(pred_targets) # numpy obj to tensor
        test_targets = torch.tensor(test_targets)
        test_accuracy = accuracy_score(test_targets, pred_targets)
        print('Classification Report:\n', classification_report(test_targets, pred_targets, digits=4))
    print('test accuracy: ',test_accuracy)
    total_test_step += 1
    pred_y_auc = torch.cat(pred_y_auc, dim=0) # tensor([-19, -29, -40])
    pred_y_auc = softmax(torch.as_tensor(pred_y_auc)) # tensor([[0.9,0.05,0.05],[...]])
    if target_num == 2:
        pred_y_auc = pred_y_auc[:,1]

    pred_y_auc = pred_y_auc.cpu()
    test_AUC = roc_auc_score(test_targets, pred_y_auc)
    print('AUC score: \n', test_AUC)
    writer.add_scalar("test accuracy", test_accuracy, total_test_step)
    writer.add_scalar("test AUC", test_AUC, total_test_step)

    
# test explainable AI
shap_data = train_X[:10,:]
print('shap_data:\n', type(shap_data), shap_data)
print(shap_data.size())
explainer = shap.DeepExplainer(finmodel, shap_data)
shap_values = explainer.shap_values(shap_data)
print('shapley values:\n', shap_values)

# summary plot
shap.summary_plot(shap_values, shap_data, feature_names=column_names)
plt.savefig("./codes/XAI/imgs/shap_summary_plot.png",dpi=500, bbox_inches='tight')

# decision plot
# expected_values = explainer.expected_value
# print(type(expected_values), expected_values)
# print(type(shap_values), shap_values.shape(), shap_values)
# shap.multioutput_decision_plot(list(expected_values), shap_values, row_index=1, feature_names=column_names)
# plt.savefig("./codes/XAI/imgs/shap_decision_plot.png",dpi=500, bbox_inches='tight')

writer.close()
