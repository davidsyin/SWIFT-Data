import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment

from sklearn.metrics import confusion_matrix            
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        print("Test loader")
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        print("Threshold loader")
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        # print(self.thre_loader[0])
        # print("Test loader", len(self.test_loader))
        print("Threshold loader", len(self.thre_loader))

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            # series_loss = 0.0
            # prior_loss = 0.0
            # for u in range(len(prior)):
            #     series_loss += (torch.mean(my_kl_loss(series[u], (
            #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                    self.win_size)).detach())) + torch.mean(
            #         my_kl_loss(
            #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                     self.win_size)).detach(),
            #             series[u]))) #Optimize series, prior is detached
            #     prior_loss += (torch.mean(
            #         my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                            self.win_size)),
            #                    series[u].detach())) + torch.mean(
            #         my_kl_loss(series[u].detach(),
            #                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       # self.win_size))))) #Optimize prior, series is detached
            def _flatten_layers_scales(nested_list):
                flat = []
                for item in nested_list:
                    if isinstance(item, (list, tuple)):
                        flat.extend(item)
                    elif item is not None:
                        flat.append(item)
                return flat
            series_all = _flatten_layers_scales(series)   # each: [B, H, L, L], softmax probs
            prior_all  = _flatten_layers_scales(prior)    # each: [B, H, L, L], unnormalized Gaussian
        
            # safety check: keep same count
            assert len(series_all) == len(prior_all) and len(prior_all) > 0, "series/prior shapes mismatch"
        
            # ---- Association discrepancy (averaged over all layers & scales) ----
            series_loss = 0.0
            prior_loss  = 0.0
            eps = 1e-12
        
            for s, p in zip(series_all, prior_all):
                # Normalize prior along last dim to make a proper distribution
                p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)          # [B,H,L,L]
        
                # KL(series || prior) + KL(prior || series) with appropriate stop-grads
                # (same minimax structure as original code)
                series_loss += (
                    torch.mean(my_kl_loss(s,        p_norm.detach()))
                  + torch.mean(my_kl_loss(p_norm.detach(), s))
                )
        
                prior_loss  += (
                    torch.mean(my_kl_loss(p_norm,   s.detach()))
                  + torch.mean(my_kl_loss(s.detach(), p_norm))
                )
        
            count = float(len(prior_all))
            series_loss = series_loss / count
            prior_loss  = prior_loss  / count
            # series_loss = series_loss / len(prior)
            # prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item()) #Make series (attention matrix) as different from prior (expected attention matrix) as possible. Unless it is an anomaly, moving series away from prior would make rec_loss too large.
            loss_2.append((rec_loss + self.k * prior_loss).item()) #Makes prior as close to series as possible (anomalies have a non-gaussian attention matrix)
            if i > 10:
                break

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")
        do_mod_attn = True

        time_now = time.time()
        # self.model.load_state_dict(
        #     torch.load(
        #         "../turbine_checkpoints/turbine_1.pth"))
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        def _flatten_layers_scales(nested_list):
            flat = []
            for item in nested_list:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                elif item is not None:
                    flat.append(item)
            return flat

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)
                series_loss = 0.0
                prior_loss = 0.0
                if not do_mod_attn:
                    # calculate Association discrepancy
                    for u in range(len(prior)):
                        series_loss += (torch.mean(my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach())) + torch.mean(
                            my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               self.win_size)).detach(),
                                       series[u])))
                        prior_loss += (torch.mean(my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach())) + torch.mean(
                            my_kl_loss(series[u].detach(), (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)))))
    
                    series_loss = series_loss / len(prior)
                    prior_loss = prior_loss / len(prior)
                else:
                    series_all = _flatten_layers_scales(series)   # each: [B, H, L, L], softmax probs
                    prior_all  = _flatten_layers_scales(prior)    # each: [B, H, L, L], unnormalized Gaussian
                
                    # safety check: keep same count
                    assert len(series_all) == len(prior_all) and len(prior_all) > 0, "series/prior shapes mismatch"
                
                    # ---- Association discrepancy (averaged over all layers & scales) ----
                    eps = 1e-12
                
                    for s, p in zip(series_all, prior_all):
                        # Normalize prior along last dim to make a proper distribution
                        p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)          # [B,H,L,L]
                
                        # KL(series || prior) + KL(prior || series) with appropriate stop-grads
                        # (same minimax structure as original code)
                        series_loss += (
                            torch.mean(my_kl_loss(s,        p_norm.detach()))
                          + torch.mean(my_kl_loss(p_norm.detach(), s))
                        )
                
                        prior_loss  += (
                            torch.mean(my_kl_loss(p_norm,   s.detach()))
                          + torch.mean(my_kl_loss(s.detach(), p_norm))
                        )
                
                    count = float(len(prior_all))
                    series_loss = series_loss / count
                    prior_loss  = prior_loss  / count

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if (i+1) % 10000 == 0:
                    train_loss = np.average(loss1_list)
                    vali_loss1, vali_loss2 = self.vali(self.test_loader)
                    print(f"{i+1} iterations, train loss = {train_loss}, vali_loss = {vali_loss1}, {vali_loss2}")
                    torch.save(self.model.state_dict(), os.path.join(path,  f'../{i+1}_checkpoint_reducedlr.pth'))
                    
                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3).to("cuda")
        self.model.load_state_dict(
            torch.load(
                "../turbine_checkpoints/Custom_checkpoint.pth"))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)
        do_mod_attn = False

        def _flatten_layers_scales(nested_list):
            flat = []
            for item in nested_list:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                elif item is not None:
                    flat.append(item)
            return flat
            
        def infer(
            data_loader,                              # an iterable of (input, labels or None)
            threshold: float = None,                  # optional absolute threshold on the energy
            percentile: float = None,                 # OR choose a percentile on provided calibration_scores
            calibration_scores: np.ndarray = None,    # optional array to compute percentile threshold
            temperature: float = 50.0,
            do_mod_attn: bool = False,                # True if your model returns per-scale lists
        ):
            """
            Returns:
              scores: 1D np.ndarray of per-timestep anomaly scores (concatenated over batches)
              preds:  1D np.ndarray of {0,1} if threshold/percentile provided, else None
            """
            # ----- load model on the right device -----
        
            criterion = nn.MSELoss(reduction='none')
        
            @torch.no_grad()
            def _flatten_layers_scales(nested):
                flat = []
                for item in nested:
                    if isinstance(item, (list, tuple)):
                        flat.extend(item)
                    elif item is not None:
                        flat.append(item)
                return flat
        
            @torch.no_grad()
            def _discrepancy(series, prior):
                """
                Discrepancy scalar (averaged over all (layer,scale) pairs).
                Each s/p tensor: [B, H, L, L]; prior is unnormalized.
                """
                eps = 1e-12
                # flatten when multi-scale returns lists inside each layer
                if len(series) > 0 and isinstance(series[0], (list, tuple)):
                    series_all = _flatten_layers_scales(series)
                    prior_all  = _flatten_layers_scales(prior)
                else:
                    series_all = list(series)
                    prior_all  = list(prior)
        
                assert len(series_all) == len(prior_all) > 0, "series/prior mismatch"
        
                s_loss = 0.0
                p_loss = 0.0
                for s, p in zip(series_all, prior_all):
                    p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)
                    s_loss += (torch.mean(my_kl_loss(s, p_norm.detach())) +
                               torch.mean(my_kl_loss(p_norm.detach(), s)))
                    p_loss += (torch.mean(my_kl_loss(p_norm, s.detach())) +
                               torch.mean(my_kl_loss(s.detach(), p_norm)))
        
                count = float(len(prior_all))
                s_loss = (s_loss / count) * temperature
                p_loss = (p_loss / count) * temperature
                return s_loss, p_loss  # scalars (0-D tensors)
                
            scores = []
            print("Computing anomaly scores")
            with torch.no_grad():
                print("Loader length", len(data_loader))
                for i, batch in enumerate(data_loader):
                    # print("batch", batch)
                    # print("Length:", len(batch))
                    # print("Num features:", len(batch[0]))
                    # if i%2000 != 0:
                    #     continue
                        
                    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                        x = batch[0]
                    else:
                        x = batch
                    x = x.float().to(self.device)                  # [B, L, C], input
        
                    output, series, prior, _ = self.model(x)       # output: [B, L, C]
                    # reconstruction energy per time step
                    rec = torch.mean(criterion(x, output), dim=-1) # [B, L], loss
        
                    # anomaly discrepancy gate (scalar)
                    if do_mod_attn:
                        s_loss, p_loss = _discrepancy(series, prior)
                    else:
                        s_loss, p_loss = _discrepancy(series, prior)

                    # print("loss", s_loss, p_loss)
                    # gate = torch.exp(-(s_loss + p_loss))           # scalar
                    gate = torch.softmax((-s_loss - p_loss), dim=-1)
                    # print("gate", gate)
                    # print("rec", rec)
                    energy = (gate * rec)[:,0]                            # broadcast to [B, L]
                    if (i%1000 == 0):
                        print("Example energy: ", energy, energy.shape)
                    scores.append(energy.detach().cpu().numpy())
        
            scores = np.concatenate(scores, axis=0).reshape(-1)    # Turn back into 1D - no more batches
            # print("scores", scores, len(scores))
            # preds = None
            # if percentile is not None:
            #     assert calibration_scores is not None and calibration_scores.size > 0, \
            #         "Provide calibration_scores to compute a percentile threshold."
            #     thresh = np.percentile(calibration_scores, percentile)
            #     preds = (scores > thresh).astype(int)
            # elif threshold is not None:
            #     preds = (scores > float(threshold)).astype(int)
        
            return scores#, preds


        
        # # train_scores = infer(self.train_loader, temperature=50.0)

        # 2) Test scores and labels
        print("Threshold loader length", len(self.thre_loader))
        test_scores = infer(self.thre_loader, temperature=50.0)
        test_labels = []

        i = 0
        for _, labels in self.thre_loader:
            # if i%2000 != 0:
            #     i += 1
            #     continue
            # print(i)
            # if i == 10000:
            #     break
            # print(labels.numpy())
            # break
            test_labels.append(labels.numpy()[0][0])
            i += 1
        test_labels = np.array(test_labels)
        print("Number of labels appended:", i)
        print(test_labels)
        # test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

        print("test_labels", test_labels, sum(test_labels), len(test_labels))
        np.save("test_labels_turbine.npy", test_labels)
        
        print("test_scores", test_scores, np.sum(test_scores), len(test_scores))
        np.save("test_scores_turbine.npy", test_scores)

        
        # 3) Sweep thresholds
        # test_scores = np.load("test_scores_turbine.npy")
        # test_labels = np.load("test_labels_turbine.npy")
        combined_scores = np.concatenate([test_scores])
        # combined_scores = np.concatenate([train_scores, test_scores])
        # print(np.sum(combined_scores))
        # print(np.sum(test_labels))
        print(test_labels.shape, test_scores.shape)
        
        for anomaly_ratio in (np.linspace(50, 100, 50)):
            # thresh = np.percentile(np.concatenate([train_scores, test_scores]), 100 - anomaly_ratio)
            thresh = np.percentile(combined_scores, 100 - anomaly_ratio)
            pred = (test_scores > thresh).astype(int)
            assert len(pred) == len(test_labels)
            
            tp = 0 # positive, predicted positive
            tn = 0 # negative, predicted negative
            fp = 0 # negative, predicted positive
            fn = 0 # positive, predicted negative
            # print(sum(test_labels))
            print("len(pred)", len(pred))
            for i in range(len(pred)):
                if test_labels[i] == 1:
                    if pred[i] == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if pred[i] == 1:
                        fp += 1
                    else:
                        tn += 1
            print("Anomaly ratio:", anomaly_ratio)
            print("TP:", tp)
            print("FP:", fp)
            print("TN:", tn)
            print("FN:", fn)
            print("Accuracy:", (tp+tn)/(tp+tn+fp+fn))
            print("Precision: ", tp/(tp+fp))
            print("Recall: ", tp/(tp+fn))
            print("F-score: ", 2*tp/(2*tp+fp+fn))
            print("\n")
            # with open('results_reducedlr/confusionmatrix_3000winsize.txt', 'w') as f:
            #     f.write(f"Anomaly ratio: {anomaly_ratio}\n")
            #     f.write(f"TP: {tp}\n")
            #     f.write(f"FP: {fp}\n")
            #     f.write(f"TN: {tn}\n")
            #     f.write(f"FN: {fn}\n")
            #     f.write("\n")

            
        # for i, (input_data, labels) in enumerate(self.train_loader):
        #     if i == 2000:
        #         break
        #     if (i%100 == 0):
        #         print(f"{i} train examples processed")
        #     input = input_data.float().to(self.device)
        #     output, series, prior, _ = self.model(input)
        #     loss = torch.mean(criterion(input, output), dim=-1)
            
        #     series_loss = 0.0
        #     prior_loss = 0.0
        #     eps = 1e-12
            
        #     series_all = _flatten_layers_scales(series)   # each: [B, H, L, L], softmax probs
        #     prior_all  = _flatten_layers_scales(prior)    # each: [B, H, L, L], unnormalized Gaussian
        
        #     # safety check: keep same count
        #     assert len(series_all) == len(prior_all) and len(prior_all) > 0, "series/prior shapes mismatch"
        #     for u in range(len(prior)):
        #         if do_mod_attn:
        #             if u == 0:          
        #                 for s, p in zip(series_all, prior_all):
        #                     # Normalize prior along last dim to make a proper distribution
        #                     p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)          # [B,H,L,L]
                    
        #                     # KL(series || prior) + KL(prior || series) with appropriate stop-grads
        #                     # (same minimax structure as original code)
        #                     series_loss += (
        #                         torch.mean(my_kl_loss(s,        p_norm.detach()))
        #                       + torch.mean(my_kl_loss(p_norm.detach(), s))
        #                     ) * temperature
                    
        #                     prior_loss += (
        #                         torch.mean(my_kl_loss(p_norm,   s.detach()))
        #                       + torch.mean(my_kl_loss(s.detach(), p_norm))
        #                     ) * temperature
        #             else:
        #                 for s, p in zip(series_all, prior_all):
        #                     # Normalize prior along last dim to make a proper distribution
        #                     p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)          # [B,H,L,L]
                    
        #                     # KL(series || prior) + KL(prior || series) with appropriate stop-grads
        #                     # (same minimax structure as original code)
        #                     series_loss += (
        #                         torch.mean(my_kl_loss(s,        p_norm.detach()))
        #                       + torch.mean(my_kl_loss(p_norm.detach(), s))
        #                     ) * temperature
                    
        #                     prior_loss += (
        #                         torch.mean(my_kl_loss(p_norm,   s.detach()))
        #                       + torch.mean(my_kl_loss(s.detach(), p_norm))
        #                     ) * temperature
        #         else:
        #             if u == 0:
        #                 series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat
        #                     (1, 1, 1, self.win_size)).detach()) * temperature
        #                 prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat
        #                     (1, 1, 1, self.win_size)), series[u].detach()) * temperature
                        
        #             else:
        #                 series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
        #                     .repeat(1, 1, 1, self.win_size)).detach()) * temperature
        #                 prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
        #                     .repeat(1, 1, 1, self.win_size)), series[u].detach()) * temperature
    
        #         count = float(len(prior_all))
        #         series_loss = series_loss / count
        #         prior_loss  = prior_loss  / count
            
        #     metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        #     cri = metric * loss
        #     cri = cri.detach().cpu().numpy()
        #     attens_energy.append(cri)
        #     # attens_energy.append(0)
        #     # print(attens_energy)
        # # print(attens_energy)
        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # train_energy = np.array(attens_energy)

        # # (2) find the threshold
        # # max_test_exs = 3000
        
        # attens_energy = []
        # print(len(self.thre_loader))
        # for i, (input_data, labels) in enumerate(self.thre_loader):
        #     # if i == max_test_exs:
        #     #     break
        #     if (i%100 == 0):
        #         print(f"{i} threshold examples processed")
        #     input = input_data.float().to(self.device)
        #     output, series, prior, _ = self.model(input)
        #     series_all = _flatten_layers_scales(series)   # each: [B, H, L, L], softmax probs
        #     prior_all  = _flatten_layers_scales(prior)    # each: [B, H, L, L], unnormalized Gaussian

        #     loss = torch.mean(criterion(input, output), dim=-1)

        #     series_loss = 0.0
        #     prior_loss = 0.0
        #     for u in range(len(prior)):
        #         if do_mod_attn:
        #             if u == 0:          
        #                 for s, p in zip(series_all, prior_all):
        #                     # Normalize prior along last dim to make a proper distribution
        #                     p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)          # [B,H,L,L]
                    
        #                     # KL(series || prior) + KL(prior || series) with appropriate stop-grads
        #                     # (same minimax structure as original code)
        #                     series_loss += (
        #                         torch.mean(my_kl_loss(s,        p_norm.detach()))
        #                       + torch.mean(my_kl_loss(p_norm.detach(), s))
        #                     ) * temperature
                    
        #                     prior_loss += (
        #                         torch.mean(my_kl_loss(p_norm,   s.detach()))
        #                       + torch.mean(my_kl_loss(s.detach(), p_norm))
        #                     ) * temperature
        #             else:
        #                 for s, p in zip(series_all, prior_all):
        #                     # Normalize prior along last dim to make a proper distribution
        #                     p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)          # [B,H,L,L]
                    
        #                     # KL(series || prior) + KL(prior || series) with appropriate stop-grads
        #                     # (same minimax structure as original code)
        #                     series_loss += (
        #                         torch.mean(my_kl_loss(s,        p_norm.detach()))
        #                       + torch.mean(my_kl_loss(p_norm.detach(), s))
        #                     ) * temperature
                    
        #                     prior_loss += (
        #                         torch.mean(my_kl_loss(p_norm,   s.detach()))
        #                       + torch.mean(my_kl_loss(s.detach(), p_norm))
        #                     ) * temperature
        #         else:
        #             if u == 0:
        #                 series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat
        #                     (1, 1, 1, self.win_size)).detach()) * temperature
        #                 prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat
        #                     (1, 1, 1, self.win_size)), series[u].detach()) * temperature
                        
        #             else:
        #                 series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
        #                     .repeat(1, 1, 1, self.win_size)).detach()) * temperature
        #                 prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
        #                     .repeat(1, 1, 1, self.win_size)), series[u].detach()) * temperature
    
        #         count = float(len(prior_all))
        #         series_loss = series_loss / count
        #         prior_loss  = prior_loss  / count

        #     # Metric
        #     metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        #     cri = metric * loss
        #     cri = cri.detach().cpu().numpy()
        #     attens_energy.append(cri)
        #     # attens_energy.append(0)
            
        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # test_energy = np.array(attens_energy)
        # combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        # np.save("combined_energy_reducedlr.npy", combined_energy)
        
        # # (3) evaluation on the test set
        # test_labels = []
        # attens_energy = []
        # labels_len = 0
        
        # print("Evaluation started")
        # for i, (input_data, labels) in enumerate(self.thre_loader):
        #     # if i == max_test_exs:
        #     #     break
        #     if i == 0:
        #         labels_len = len(labels)
        #     if i%1000 == 0:
        #         print(f"{i} evaluation examples processed")
        #     input = input_data.float().to(self.device)
        #     output, series, prior, _ = self.model(input)
        #     series_all = _flatten_layers_scales(series)   # each: [B, H, L, L], softmax probs
        #     prior_all  = _flatten_layers_scales(prior)    # each: [B, H, L, L], unnormalized Gaussian

        #     loss = torch.mean(criterion(input, output), dim=-1)

        #     series_loss = 0.0
        #     prior_loss = 0.0
        #     for u in range(len(prior)):
        #         if not do_mod_attn:
        #             if u == 0:          
        #                 for s, p in zip(series_all, prior_all):
        #                     # Normalize prior along last dim to make a proper distribution
        #                     p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)          # [B,H,L,L]
                    
        #                     # KL(series || prior) + KL(prior || series) with appropriate stop-grads
        #                     # (same minimax structure as original code)
        #                     series_loss += (
        #                         torch.mean(my_kl_loss(s,        p_norm.detach()))
        #                       + torch.mean(my_kl_loss(p_norm.detach(), s))
        #                     ) * temperature
                    
        #                     prior_loss += (
        #                         torch.mean(my_kl_loss(p_norm,   s.detach()))
        #                       + torch.mean(my_kl_loss(s.detach(), p_norm))
        #                     ) * temperature
        #             else:
        #                 for s, p in zip(series_all, prior_all):
        #                     # Normalize prior along last dim to make a proper distribution
        #                     p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)          # [B,H,L,L]
                    
        #                     # KL(series || prior) + KL(prior || series) with appropriate stop-grads
        #                     # (same minimax structure as original code)
        #                     series_loss += (
        #                         torch.mean(my_kl_loss(s,        p_norm.detach()))
        #                       + torch.mean(my_kl_loss(p_norm.detach(), s))
        #                     ) * temperature
                    
        #                     prior_loss += (
        #                         torch.mean(my_kl_loss(p_norm,   s.detach()))
        #                       + torch.mean(my_kl_loss(s.detach(), p_norm))
        #                     ) * temperature
        #         else:
        #             if u == 0:
        #                 series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat
        #                     (1, 1, 1, self.win_size)).detach()) * temperature
        #                 prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat
        #                     (1, 1, 1, self.win_size)), series[u].detach()) * temperature
                        
        #             else:
        #                 series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
        #                     .repeat(1, 1, 1, self.win_size)).detach()) * temperature
        #                 prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
        #                     .repeat(1, 1, 1, self.win_size)), series[u].detach()) * temperature
        #     metric = torch.softmax((-series_loss - prior_loss), dim=-1)

        #     cri = metric * loss
        #     cri = cri.detach().cpu().numpy()
        #     if len(labels) != labels_len:
        #         continue
        #     attens_energy.append(cri)
        #     test_labels.append(labels)
        #     # attens_energy.append(0)
        #     # test_labels.append(labels)

        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        # test_energy = np.array(attens_energy)
        # test_labels = np.array(test_labels)

        # np.save("test_energy_reducedlr.npy", test_energy)
        # np.save("test_labels_reducedlr.npy", test_labels)
        # # test_energy = np.load("test_energy.npy")
        # # test_labels = np.load("test_labels.npy")
        # # combined_energy = np.load("combined_energy.npy")
        
        # for anomaly_ratio in (np.linspace(0.1,1,50)**2 / 10):
        #     thresh = np.percentile(combined_energy, 100 - anomaly_ratio)
        #     print("Threshold :", thresh)
            
        #     pred = (test_energy > thresh).astype(int)
        #     gt = test_labels.astype(int) #ground truth
        #     # gt = gt[:max_test_exs]
        #     print("Total pred, total gt", sum(pred), sum(gt))
        #     # print("pred:   ", pred.shape)
        #     # print("gt:     ", gt.shape)
    
        #     # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        #     anomaly_state = False
        #     for i in range(len(gt)):
        #         if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
        #             anomaly_state = True
        #             for j in range(i, 0, -1):
        #                 if gt[j] == 0:
        #                     break
        #                 else:
        #                     if pred[j] == 0:
        #                         pred[j] = 1
        #             for j in range(i, len(gt)):
        #                 if gt[j] == 0:
        #                     break
        #                 else:
        #                     if pred[j] == 0:
        #                         pred[j] = 1
        #         elif gt[i] == 0:
        #             anomaly_state = False
        #         if anomaly_state:
        #             pred[i] = 1
    
        #     pred = np.array(pred)
        #     gt = np.array(gt)
        #     # print("pred: ", pred.shape)
        #     # print("gt:   ", gt.shape)
    
            
            
        #     accuracy = accuracy_score(gt, pred)
        #     precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
        #                                                                       average='binary')
        #     cm = confusion_matrix(gt, pred)
        #     print(f"Anomaly ratio: {anomaly_ratio}, Confusion matrix: {cm}")
        #     print(
        #         "Anomaly ratio: {:0.8f}, Accuracy : {:0.8f}, Precision : {:0.8f}, Recall : {:0.8f}, F-score : {:0.8f} ".format(
        #             anomaly_ratio, accuracy, precision,
        #             recall, f_score))

        # return accuracy, precision, recall, f_score
