# test centralised learning with Differential Privacy enabled.

from cgi import test
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
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
import codes.lib.rdp_accountant as rdp_accountant
from timeit import default_timer as timer

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
            # nn.Linear(64, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, outdims),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.finmodel(x)
        return output
softmax = torch.nn.Softmax(dim=1)
target = 'personal_loan'

selected_variables = ['age','experience','income','family','ccavg','education','mortgage','securities_account','cd_account','online','creditcard','personal_loan']
categorical_variables = ['family','education','securities_account','cd_account','online','creditcard','personal_loan']
categorical_variables.append(categorical_variables.pop(categorical_variables.index(target))) # move the target column to the end
print(categorical_variables)
continuous_variables = [col for col in selected_variables if col not in categorical_variables]
print(continuous_variables)

orig_data = pd.read_csv('./sourceData/bank_Loan.csv', usecols=selected_variables)

target_num = orig_data[target].nunique()
print('Output dimension: ', target_num)
print('Classes to predict:\n', orig_data[target].value_counts())
# preprocessing, onehot, minmaxscale
encoded_data = pd.get_dummies(orig_data, prefix=categorical_variables, columns=categorical_variables, prefix_sep='_')
scaler = MinMaxScaler(feature_range=(-1, 1))
encoded_data[continuous_variables] = scaler.fit_transform(encoded_data[continuous_variables])

print(encoded_data.head())
print(encoded_data.shape)
print('encoded data shape: ', encoded_data.shape)
trainSet = encoded_data.sample(frac=0.8, random_state=64)
testSet = encoded_data.drop(trainSet.index)
train_X = trainSet.iloc[:, :-target_num]
train_y = trainSet.iloc[:, -target_num:]
print(train_X.head())
print(train_y.head())
test_X = testSet.iloc[:, :-target_num]
test_y = testSet.iloc[:, -target_num:]
print('train_X', train_X.shape)
print('test_X', test_X.shape)
print('train_y', train_y.shape)
print('test_y', test_y.shape)
# record logs
writer = SummaryWriter('./logs/log_train/dp_disabled/adam/adam_lightmodel')
total_train_step = 0
total_test_step = 0
total_test_loss = 0
# hyperparameters of finmodel
finmodel = FintrustML(dims=train_X.shape[1], outdims=train_y.shape[1]).to(device)

loss_fn = nn.BCELoss() # ! for binary classification
optimizer = torch.optim.Adam(finmodel.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(finmodel.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # dynamic lr
batch_size = 32
steps_per_epoch = round(train_X.shape[0]/batch_size)
rounds = 1000

################################ Differential Privacy #################################

micro_batch_size = batch_size
clip_coeff = 1
sigma = 1.9
target_delta = 1e-5
max_lmbd = 4095
Diff_Privacy = False
calc_epsilon = False

#######################################################################################
start = timer()
# start training:
for i in range(rounds):
    print("-----------start training round {}------------".format(i+1))
    finmodel.train()
    for j in range(steps_per_epoch):
        optimizer.zero_grad()
        features = train_X.sample(batch_size)
        targets = train_y.loc[features.index]
        features = torch.tensor(features.values, dtype=torch.float).to(device)
        targets = torch.tensor(targets.values, dtype=torch.float).to(device)
        outputs = finmodel(features) # [batch_size, train_y.shape[1]]
        # print('output', outputs[0])
        # targets = torch.argmax(targets, dim=1) # 1D array
        loss = loss_fn(outputs, targets)
        loss.backward()

        ######################### Differential Privacy ################################

        if Diff_Privacy == True:
            clipped_grads = {name: torch.zeros_like(param) for name, param in finmodel.named_parameters()}
            # Gradient clipping ensures the gradient vector g has norm at most equal to threshold. 
            torch.nn.utils.clip_grad_norm_(finmodel.parameters(), clip_coeff)
            for name, param in finmodel.named_parameters():
                clipped_grads[name] += param.grad
            finmodel.zero_grad()

            for name, param in finmodel.named_parameters():
                param.grad = (clipped_grads[name] + torch.FloatTensor(clipped_grads[name].size()).normal_(0, sigma * clip_coeff).cuda()) / (train_X.shape[0] /micro_batch_size)

        ###############################################################################

        optimizer.step()
        total_train_step += 1
        if j == steps_per_epoch-1:
            print("Loss of train step {}: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train loss", loss.item(), i)

    ######################### Differential Privacy #####################################
    steps = total_train_step
    if Diff_Privacy == True & calc_epsilon == True:
        lmbds = range(2, max_lmbd +1)
        rdp = rdp_accountant.compute_rdp(micro_batch_size / train_X.shape[0], sigma, steps, lmbds)
        epsilon, _, _ = rdp_accountant.get_privacy_spent(lmbds, rdp, target_delta=1e-5)

        log_mlp_dp = pd.DataFrame([[i, steps, loss.item(), epsilon]])

        writer.add_scalar("epsilon", epsilon, i)
        if i == 0:
            log_mlp_dp.to_csv("./outputs/dp/log_mlp_dp.csv", mode="a", header=["Round", "steps", "train_loss", "epsilon"], index=False)
        else:
            log_mlp_dp.to_csv("./outputs/dp/log_mlp_dp.csv", mode="a", header=False, index=False)

    ####################################################################################
    # scheduler.step()

    # start testing:
    finmodel.eval()
    with torch.no_grad():
        test_features = torch.tensor(test_X.values, dtype=torch.float).to(device)
        test_targets = torch.tensor(test_y.values, dtype=torch.float).to(device)
        test_targets = torch.argmax(test_targets, dim=1)
        pred_targets = []
        pred_y_auc = []
        for k in range(test_features.shape[0]):
            output = finmodel(test_features[k, :])
            mask = torch.argmax(output)
            mask = torch.reshape(mask, (-1,))
            pred_targets.extend(mask)
            pred_y_auc.append(output)
        pred_targets = torch.tensor(pred_targets)
        test_targets = test_targets.cpu() # cpu needed
        test_accuracy = accuracy_score(test_targets, pred_targets)
        pred_y_auc = torch.stack([i for i in pred_y_auc]) # tensor([-19, -29, -40])
        pred_y_auc = softmax(torch.as_tensor(pred_y_auc)) # tensor([[0.9,0.05,0.05],[...]])
        pred_y_auc = pred_y_auc[:,1]
        pred_y_auc = pred_y_auc.cpu()
        print('Classification Report:\n', classification_report(test_targets, pred_targets, digits=4))
    print('test accuracy: ',test_accuracy)
    total_test_step += 1
    print('AUC score: \n', roc_auc_score(test_targets, pred_y_auc))
    writer.add_scalar("test accuracy", test_accuracy, total_test_step)
end = timer()

print('Total time elapsed', end - start)

writer.close()
