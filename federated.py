import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import load_data_sc, load_data_sc_transport
import argparse
import numpy as np
from scipy.stats import pearsonr
from mats2edges import mats2edges
from model_pytorch import msenanloss
from model_pytorch import dnn_3l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    atlas_t = args.t
    task = args.task
    sd = args.sd
    epochs = args.epochs
    weight_decay = args.weight_decay
    lr = args.lr
    nsteps = args.nsteps
    n_l1 = args.n_l1
    n_l2 = args.n_l2
    dropout = args.dropout
    pace = args.pace

    mse_loss = nn.MSELoss()
    y = np.load("hcpd_age.npy")
    index = ~np.isnan(y)
    y = y[index]
    y_mean = np.mean(y)
    y_dev = np.std(y)
    y = (y - y_mean)/y_dev
    np.random.seed(sd)
    id = np.arange(625)
    np.random.shuffle(id)
    id_test = id[:125]
    id_train = []
    id_train.append(id[125:250])
    id_train.append(id[250:375])
    id_train.append(id[375:500])
    id_train.append(id[500:625])
    atlas = ["shen", "craddock", "dosenbach", "schaefer", "brainnetome"]
    atlas.remove(atlas_t)

    ids = id_test
    comT = load_data_sc(atlas_t, task) 
    comT = comT[:,:,index]
    x_site = mats2edges(comT)[ids,:]
    y_site = y[ids]
    x_site = torch.tensor(x_site, dtype=torch.float32)
    y_site = torch.tensor(y_site.reshape(-1,1), dtype=torch.float32)
    testset = TensorDataset(x_site, y_site)
    test_loader = DataLoader(testset, batch_size=5, shuffle=False)

    models = []
    optimizers = []
    data_loaders = []
    for i in range(len(atlas)):
        atlas_s = atlas[i]
        ids = id_train[i]
        comR = load_data_sc_transport(atlas_s, atlas_t, task)
        comR = comR[:,:,index]
        x_site = mats2edges(comR)[ids,:]
        y_site = y[ids]
        x_site = torch.tensor(x_site, dtype=torch.float32)
        y_site = torch.tensor(y_site.reshape(-1,1), dtype=torch.float32)
        trainset = TensorDataset(x_site, y_site)
        train_loader = DataLoader(trainset, batch_size=len(trainset)//nsteps, shuffle=True)
        data_loaders.append(train_loader)
        net = dnn_3l(x_site.shape[1],n_l1,n_l2,dropout=dropout,output_size=1)
        net.to(device)
        models.append(net)
        optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=weight_decay)
        optimizers.append(optimizer)

    model = dnn_3l(x_site.shape[1],n_l1,n_l2,dropout=dropout,output_size=1) 
    model.to(device)

    def train(epoch):
        data_inters = [iter(train_loader) for train_loader in data_loaders]
        for i in range(4):
            models[i].train()

        #define weights
        w = dict()
        for i in range(4):
            w[i] = 0.25 
    
        loss_all = dict()
        num_data = dict()
        for i in range(4):
            loss_all[i] = 0
            num_data[i] = 0
        count = 0
        for t in range(nsteps):
            for i in range(4):
                optimizers[i].zero_grad()
                a, b = next(data_inters[i])
                num_data[i] += b.size(0)
                a = a.to(device)
                b = b.to(device)
                output = models[i](a)
                loss = mse_loss(output, b)
                loss.backward()
                loss_all[i] += loss.item() * b.size(0)
                optimizers[i].step()
            count += 1
            if count%pace ==0 or t == nsteps-1 :
                with torch.no_grad():
                    for key in model.state_dict().keys():
                        if models[0].state_dict()[key].dtype == torch.int64:
                            model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                        else:
                            temp = torch.zeros_like(model.state_dict()[key])
                            for s in range(4):
                                temp += w[s]*models[s].state_dict()[key]
                            # update global model
                            model.state_dict()[key].data.copy_(temp)
                            # update local model
                            for s in range(4):
                                models[s].state_dict()[key].data.copy_(model.state_dict()[key])
    
        return loss_all[0] / num_data[0], loss_all[1] / num_data[1], \
            loss_all[2] / num_data[2], loss_all[3] / num_data[3]


    def test(federated_model,data_loader):
        federated_model.eval()
        test_loss = 0
        outputs = []
        targets = []
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            targets.extend(target.detach().numpy())
            output = federated_model(data)
            outputs.extend(output.detach().cpu().numpy())
            test_loss += mse_loss(output, target).item()*target.size(0)

        test_loss /= len(data_loader.dataset)
        outputs = np.array(outputs)
        targets = np.array(targets)
        corr = pearsonr(targets.flatten(), outputs.flatten())
        return corr

    for epoch in range(epochs):
        print(f"Epoch Number {epoch + 1}")
        train(epoch)
        corr = test(model, test_loader)
        print(corr[0])

    np.savez("../results_fed/federated_dnn_"+atlas_t+"_"+task+"_"+str(sd)+".npz", corr=corr)


def get_args():
    '''function to get args from command line and return the args

    Returns:
        argparse.ArgumentParser: args that could be used by other function
    '''
    parser = argparse.ArgumentParser(description='federated learning')

    # general parameters
    parser.add_argument('-t', type=str, help="target atlas")
    parser.add_argument('-task', type=str, help="task")
    parser.add_argument('-sd', type=int, help="seed")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--nsteps', type=int, default=20)
    parser.add_argument('--pace', type=int, default=5)

    # hyperparameter
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_l1', type=int, default=64)
    parser.add_argument('--n_l2', type=int, default=8)

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
    
