import argparse
import yaml
import os
import glob
import numpy as np
from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
import loss as loss_factory
from wpf_dataset import PGL4WPFDataset, TestPGL4WPFDataset
from Jie import Pinjie
from metrics import regressor_detailed_scores
from utils import save_model, _create_if_not_exist, get_logger, str2bool, ensure_dir, load_model
from logging import getLogger
from tqdm import tqdm


def set_seed(seed):
    """Set seed for reproduction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_and_evaluate(config, train_data, valid_data, test_data=None):
    # weekday,time,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,Patv
    # 0       1    2    3    4    5    6    7    8    9    10   11
    name2id = {
        'weekday': 0,
        'time': 1,
        'Wspd': 2,
        'Wdir': 3,
        'Etmp': 4,
        'Itmp': 5,
        'Ndir': 6,
        'Pab1': 7,
        'Pab2': 8,
        'Pab3': 9,
        'Prtv': 10,
        'Patv': 11
    }

    # select useful features
    select = config.select
    select_ind = [name2id[name] for name in select]

    log = getLogger()

    data_mean = torch.FloatTensor(train_data.data_mean).to(config.device)  # (1, 134, 1, 1)
    data_scale = torch.FloatTensor(train_data.data_scale).to(config.device)  # (1, 134, 1, 1)

    graph = train_data.graph  # (134, 134)

    train_data_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers)

    valid_data_loader = DataLoader(
        valid_data,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers)


    if config.model == 'Pinjie':
        model = Pinjie(config=config, adj_mx=graph)
        model = model.to(config.device)
    else:
        raise ValueError('Error config.model = {}'.format(config.model))

    log.info(model)
    for name, param in model.named_parameters():
        log.info(str(name) + '\t' + str(param.shape) + '\t' +
                          str(param.device) + '\t' + str(param.requires_grad))
    total_num = sum([param.nelement() for param in model.parameters()])
    log.info('Total parameter numbers: {}'.format(total_num))

    loss_fn = getattr(loss_factory, config.loss)()

    log.info('You select `{}` optimizer.'.format(config.learner.lower()))
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    grad_accmu_steps = config.gsteps
    opt.zero_grad()

    _create_if_not_exist(config.output_path)
    global_step = 0

    best_score = np.inf
    patient = 0

    col_names = dict(
        [(v, k) for k, v in enumerate(train_data.get_raw_df()[0].columns)])

    valid_records = []
    # test_records = []

    for epoch in range(config.epoch):
        model.train()
        losses = []
        for batch_x, batch_y in tqdm(train_data_loader, 'train'):


            # weekday,time,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,Patv
            # 0       1    2    3    4    5    6    7    8    9    10   11
            if config.only_useful:
                batch_x = batch_x[:, :, :, select_ind]
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            input_y = batch_y  # (B,N,T,F)
            batch_y = batch_y[:, :, :, -1]  # (B,N,T)
            batch_y = (batch_y - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1]
            pred_y = model(batch_x, input_y, data_mean, data_scale)  # (B,N,T)
            loss = loss_fn(pred_y, batch_y, input_y, col_names)
            loss = loss / grad_accmu_steps
            # torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            loss.backward(retain_graph=True)

            if global_step % grad_accmu_steps == 0:
                opt.step()
                opt.zero_grad()
            global_step += 1
            losses.append(loss.item())
            if global_step % config.log_per_steps == 0:
                log.info("Step %s Train Loss: %s" % (global_step, loss.item()))
        log.info("Epoch=%s, exp_id=%s, Train Loss: %s" % (epoch, config.exp_id, np.mean(losses)))

        valid_r = evaluate(
                valid_data_loader,
                valid_data.get_raw_df(),
                model,
                loss_fn,
                config,
                data_mean,
                data_scale,
                tag="val",
                select_ind=select_ind)
        valid_records.append(valid_r)

        log.info("Epoch={}, exp_id={}, Valid ".format(epoch, config.exp_id) + str(dict(valid_r)))

        best_score = min(valid_r['mae'], best_score)

        if best_score == valid_r['mae']:
            patient = 0
            save_model(config.output_path+config.exp_id+'_'+config.model, model, opt=opt, steps=epoch, log=log)
        else:
            patient += 1
            if patient > config.patient:
                break

    best_epochs = min(enumerate(valid_records), key=lambda x: x[1]["mae"])[0]
    log.info("Best valid Epoch %s" % best_epochs)
    log.info("Best valid score %s" % valid_records[best_epochs])
    # log.info("Best valid test-score %s" % test_records[best_epochs])


def evaluate(valid_data_loader,
             valid_raw_df,
             model,
             loss_fn,
             config,
             data_mean,
             data_scale,
             tag="train",
             select_ind=None):
    with torch.no_grad():
        col_names = dict([(v, k) for k, v in enumerate(valid_raw_df[0].columns)])
        model.eval()
        step = 0
        pred_batch = []
        gold_batch = []
        input_batch = []
        losses = []
        for batch_x, batch_y in tqdm(valid_data_loader, tag):
            # weekday,time,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,Patv
            # 0       1    2    3    4    5    6    7    8    9    10   11
            if config.only_useful:
                batch_x = batch_x[:, :, :, select_ind]
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            pred_y = model(batch_x, batch_y, data_mean, data_scale)

            scaled_batch_y = batch_y[:, :, :, -1]  # (B,N,T)
            scaled_batch_y = (scaled_batch_y - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1]
            loss = loss_fn(pred_y, scaled_batch_y, batch_y, col_names)
            losses.append(loss.item())

            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
            pred_y = pred_y.cpu().numpy()  # (B,N,T)

            batch_y = batch_y[:, :, :, -1].cpu().numpy()  # (B,N,T)
            input_batch.append(batch_x[:, :, :, -1].cpu().numpy())  # (B,N,T)
            pred_batch.append(pred_y)
            gold_batch.append(batch_y)

            step += 1
        model.train()

        pred_batch = np.concatenate(pred_batch, axis=0)  # (B,N,T)
        gold_batch = np.concatenate(gold_batch, axis=0)  # (B,N,T)
        input_batch = np.concatenate(input_batch, axis=0)  # (B,N,T)

        pred_batch = np.expand_dims(pred_batch, -1)  # (B,N,T,1)
        gold_batch = np.expand_dims(gold_batch, -1)  # (B,N,T,1)
        input_batch = np.expand_dims(input_batch, -1)  # (B,N,T,1)

        pred_batch = np.transpose(pred_batch, [1, 0, 2, 3])  # (N,B,T,1)
        gold_batch = np.transpose(gold_batch, [1, 0, 2, 3])  # (N,B,T,1)
        input_batch = np.transpose(input_batch, [1, 0, 2, 3])  # (N,B,T,1)

        _mae, _rmse = regressor_detailed_scores(pred_batch, gold_batch,
                                                valid_raw_df, config.capacity,
                                                config.output_len)

        output_metric = {
            'mae': _mae,
            'rmse': _rmse,
            'loss': np.mean(losses),
        }

        return output_metric



def predict(config, train_data):
    log = getLogger()
    name2id = {
        'weekday': 0,
        'time': 1,
        'Wspd': 2,
        'Wdir': 3,
        'Etmp': 4,
        'Itmp': 5,
        'Ndir': 6,
        'Pab1': 7,
        'Pab2': 8,
        'Pab3': 9,
        'Prtv': 10,
        'Patv': 11
    }
    select = config.select
    select_ind = [name2id[name] for name in select]

    test_data_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers)

    with torch.no_grad():
        data_mean = torch.FloatTensor(train_data.data_mean).to(config.device)  # (1, 134, 1, 1)
        data_scale = torch.FloatTensor(train_data.data_scale).to(config.device)  # (1, 134, 1, 1)

        graph = train_data.graph  # (134, 134)

        if config.model == 'Pinjie':
            model = Pinjie(config=config, adj_mx=graph)
            model = model.to(config.device)
        else:
            raise ValueError('Error config.model = {}'.format(config.model))

        output_path = config.output_path+config.exp_id+'_'+config.model
        load_model(os.path.join(output_path, "model_%d.pt" % config.best), model, log=log)
        log.info("Best %s model Exp_id %s" % (config.best, config.exp_id))

        model.eval()

        test_x = sorted(glob.glob(os.path.join("./data", "test_x", "*")))
        test_y = sorted(glob.glob(os.path.join("./data", "test_y", "*")))

        maes, rmses = [], []

        for batch_x, batch_y in tqdm(test_data_loader):

            test_x_ds = batch_x
            test_y_ds = batch_y

            if config.only_useful:
                test_x = torch.FloatTensor(
                    test_x_ds[:, :, -config.input_len:, select_ind]).to(config.device)
                test_y = torch.FloatTensor(
                    test_y_ds[:, :, :config.output_len, select_ind]).to(config.device)

            pred_y = model(test_x, None, data_mean, data_scale)  # (B,N,T)
            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])

            pred_y = np.expand_dims(pred_y.cpu().numpy(), -1)  # (B,N,T,1)
            test_y = test_y[:, :, :, -1:].cpu().numpy()  # (B,N,T,F)

            pred_y = np.transpose(pred_y, [  # (N,B,T,1)
                1,
                0,
                2,
                3,
            ])
            test_y = np.transpose(test_y, [  # (N,B,T,F)
                1,
                0,
                2,
                3,
            ])
            test_y_df = train_data.get_raw_df()

            _mae, _rmse = regressor_detailed_scores(
                pred_y, test_y, test_y_df, config.capacity, config.output_len)
            print('\n\tThe prediction for File {} -- '
                  'RMSE: {}, MAE: {}'.format(batch_y, _rmse, _mae))
            maes.append(_mae)
            rmses.append(_rmse)

        avg_mae = np.array(maes).mean()
        avg_rmse = np.array(rmses).mean()
        log.info('\n --- Final MAE: {}, RMSE: {} ---'.format(avg_mae, avg_rmse))


if __name__ == "__main__":

    pro_flag = 'test'
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument("--model", type=str, default="Pinjie")
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--input_len", type=int, default=144, help='input data len')
    parser.add_argument("--output_len", type=int, default=288, help='output data len')
    parser.add_argument("--output_len_dl", type=int, default=72, )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_days", type=int, default=15)
    parser.add_argument("--val_days", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=134)

    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--gcn_out", type=int, default=32)
    parser.add_argument("--f_out", type=int, default=64)

    parser.add_argument("--residual_channels", type=int, default=32)
    parser.add_argument("--dilation_channels", type=int, default=32)
    parser.add_argument("--skip_channels", type=int, default=128)
    parser.add_argument("--end_channels", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--moving_avg", type=int, default=25)

    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--random", type=str2bool, default=False)
    # parser.add_argument("--enhance", type=str2bool, default=True, help='Whether enhance the time dim')
    parser.add_argument("--only_useful", type=str2bool, default=True, help='Whether remove some feature')
    parser.add_argument("--var_len", type=int, default=5, help='Dimensionality of input features')
    parser.add_argument("--data_diff", type=str2bool, default=False, help='Whether to use data differential features')
    parser.add_argument("--add_apt", type=str2bool, default=False, help='Whether to use adaptive matrix')
    parser.add_argument("--binary", type=str2bool, default=True, help='Whether to set the adjacency matrix as binary')
    parser.add_argument("--graph_type", type=str, default="geo", help='graph type, dtw or geo')
    parser.add_argument("--dtw_topk", type=int, default=5, help='M dtw for dtw graph')
    parser.add_argument("--weight_adj_epsilon", type=float, default=0.8, help='epsilon for geo graph')
    parser.add_argument("--gsteps", type=int, default=1, help='Gradient Accumulation')
    parser.add_argument("--loss", type=str, default='FilterHuberLoss')
    parser.add_argument("--select", nargs='+', type=str,
                        default=['weekday', 'time', 'Wspd', 'Etmp', 'Itmp', 'Prtv', 'Patv'])
    parser.add_argument("--len_select", type=int, default=7)

    parser.add_argument("--exp_id", type=str, default='24803')
    parser.add_argument("--best", type=int, default=2)
    parser.add_argument("--output_path", type=str, default='output/')

    args = parser.parse_args()
    dict_args = vars(args)

    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(dict_args)

    exp_id = config.get('exp_id', None)
    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = str(exp_id)

    logger = get_logger(config)
    logger.info(config)
    set_seed(config.seed)
    ensure_dir(config.output_path)

    train_data = PGL4WPFDataset(
        config.data_path,
        filename=config.filename,
        size=[config.input_len, config.output_len],
        flag='train',
        total_days=config.total_days,
        train_days=config.train_days,
        val_days=config.val_days,
        test_days=config.test_days,
        random=config.random,
        only_useful=config.only_useful,
        graph_type=config.graph_type,
        weight_adj_epsilon=config.weight_adj_epsilon,
        dtw_topk=config.dtw_topk,
        binary=config.binary,
    )
    valid_data = PGL4WPFDataset(
        config.data_path,
        filename=config.filename,
        size=[config.input_len, config.output_len],
        flag='val',
        total_days=config.total_days,
        train_days=config.train_days,
        val_days=config.val_days,
        test_days=config.test_days,
        only_useful=config.only_useful,
        graph_type=config.graph_type,
        weight_adj_epsilon=config.weight_adj_epsilon,
        dtw_topk=config.dtw_topk,
        binary=config.binary,
    )
    test_data = PGL4WPFDataset(
        config.data_path,
        filename=config.filename,
        size=[config.input_len, config.output_len],
        flag='test',
        total_days=config.total_days,
        train_days=config.train_days,
        val_days=config.val_days,
        test_days=config.test_days,
        only_useful=config.only_useful,
        graph_type=config.graph_type,
        weight_adj_epsilon=config.weight_adj_epsilon,
        dtw_topk=config.dtw_topk,
        binary=config.binary,
    )

    gpu_id = config.gpu_id
    if gpu_id != -1:
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')
    config['device'] = device
    if pro_flag =='train':
        train_and_evaluate(config, train_data, valid_data, None)
    else:
        predict(config, test_data)

