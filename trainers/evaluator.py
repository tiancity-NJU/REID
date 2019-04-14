# encoding: utf-8
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import gc


from utils.AlignedTripletLoss import low_memory_local_dist
from trainers.re_ranking import re_ranking as re_ranking_func

class ResNetEvaluator:
    def __init__(self, model):
        self.model = model

    def save_incorrect_pairs(self, distmat, queryloader, galleryloader, 
        g_pids, q_pids, g_camids, q_camids, savefig):
        os.makedirs(savefig, exist_ok=True)
        self.model.eval()
        m = distmat.shape[0]
        indices = np.argsort(distmat, axis=1)
        for i in range(m):
            for j in range(10):
                index = indices[i][j]
                if g_camids[index] == q_camids[i] and g_pids[index] == q_pids[i]:
                    continue
                else:
                    break
            if g_pids[index] == q_pids[i]:
                continue
            fig, axes =plt.subplots(1, 11, figsize=(12, 8))
            img = queryloader.dataset.dataset[i][0]
            img = Image.open(img).convert('RGB')
            axes[0].set_title(q_pids[i])
            axes[0].imshow(img)
            axes[0].set_axis_off()
            for j in range(10):
                gallery_index = indices[i][j]
                img = galleryloader.dataset.dataset[gallery_index][0]
                img = Image.open(img).convert('RGB')
                axes[j+1].set_title(g_pids[gallery_index])
                axes[j+1].set_axis_off()
                axes[j+1].imshow(img)
            fig.savefig(os.path.join(savefig, '%d.png' %q_pids[i]))
            plt.close(fig)

    def evaluate(self, queryloader, galleryloader, queryFliploader, galleryFliploader, 
        ranks=[1, 2, 4, 5,8, 10, 16, 20], eval_flip=False, re_ranking=False, savefig=False):
        self.model.eval()
        qf, q_pids, q_camids = [], [], []
        for inputs0, inputs1 in zip(queryloader, queryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if eval_flip:
                inputs, pids, camids = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                qf.append((feature0 + feature1) / 2.0)
            else:
                qf.append(feature0)

            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = torch.Tensor(q_pids)
        q_camids = torch.Tensor(q_camids)

        print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for inputs0, inputs1 in zip(galleryloader, galleryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if eval_flip:
                inputs, pids, camids = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                gf.append((feature0 + feature1) / 2.0)
            else:
                gf.append(feature0)
                
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = torch.Tensor(g_pids)
        g_camids = torch.Tensor(g_camids)

        print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        print("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())

        if re_ranking:
            q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
            q_q_dist.addmm_(1, -2, qf, qf.t())

            g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            g_g_dist.addmm_(1, -2, gf, gf.t())

            q_g_dist = q_g_dist.numpy()
            q_g_dist[q_g_dist < 0] = 0
            q_g_dist = np.sqrt(q_g_dist)

            q_q_dist = q_q_dist.numpy()
            q_q_dist[q_q_dist < 0] = 0
            q_q_dist = np.sqrt(q_q_dist)

            g_g_dist = g_g_dist.numpy()
            g_g_dist[g_g_dist < 0] = 0
            g_g_dist = np.sqrt(g_g_dist)

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist))
        else:
            distmat = q_g_dist 

        if savefig:
            print("Saving fingure")
            self.save_incorrect_pairs(distmat.numpy(), queryloader, galleryloader, 
                g_pids.numpy(), q_pids.numpy(), g_camids.numpy(), q_camids.numpy(), savefig)

        print("Computing CMC and mAP")
        cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
        del(qf)
        del(gf)
        try:
            gc.collect()
        except:
            print('cannot collect.....')

        return cmc[0]

    def _parse_data(self, inputs):
        imgs, pids, camids = inputs
        return imgs.cuda(), pids, camids

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature.cpu()

    def eval_func_gpu(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.size()
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        _, indices = torch.sort(distmat, dim=1)
        matches = g_pids[indices] == q_pids.view([num_q, -1]) 
        keep = ~((g_pids[indices] == q_pids.view([num_q, -1])) & (g_camids[indices]  == q_camids.view([num_q, -1])))
        #keep = g_camids[indices]  != q_camids.view([num_q, -1])

        results = []
        num_rel = []
        for i in range(num_q):
            m = matches[i][keep[i]]
            if m.any():
                num_rel.append(m.sum())
                results.append(m[:max_rank].unsqueeze(0))
        matches = torch.cat(results, dim=0).float()
        num_rel = torch.Tensor(num_rel)

        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        all_cmc = cmc.sum(dim=0) / cmc.size(0)

        pos = torch.Tensor(range(1, max_rank+1))
        temp_cmc = matches.cumsum(dim=1) / pos * matches
        AP = temp_cmc.sum(dim=1) / num_rel
        mAP = AP.sum() / AP.size(0)
        return all_cmc.numpy(), mAP.item()

    def eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP



class AlignedEvaluator:
    def __init__(self, model):
        self.model = model

    def eval_cuhk03(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank, N=100):
        """Evaluation with cuhk03 metric
        Key: one image for each gallery identity is randomly sampled for each query identity.
        Random sampling is performed N times (default: N=100).
        """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            kept_g_pids = g_pids[order][keep]
            g_pids_dict = defaultdict(list)
            for idx, pid in enumerate(kept_g_pids):
                g_pids_dict[pid].append(idx)

            cmc, AP = 0., 0.
            for repeat_idx in range(N):
                mask = np.zeros(len(orig_cmc), dtype=np.bool)
                for _, idxs in g_pids_dict.items():
                    # randomly sample one image for each gallery person
                    rnd_idx = np.random.choice(idxs)
                    mask[rnd_idx] = True
                masked_orig_cmc = orig_cmc[mask]
                _cmc = masked_orig_cmc.cumsum()
                _cmc[_cmc > 1] = 1
                cmc += _cmc[:max_rank].astype(np.float32)
                # compute AP
                num_rel = masked_orig_cmc.sum()
                tmp_cmc = masked_orig_cmc.cumsum()
                tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
                tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
                AP += tmp_cmc.sum() / num_rel
            cmc /= N
            AP /= N
            all_cmc.append(cmc)
            all_AP.append(AP)
            num_valid_q += 1.

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP

    def eval_market1501(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
        """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)

        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP




    def evaluate_mat(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False):
        if use_metric_cuhk03:
            return self.eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
        else:
            return self.eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)


    def evaluate(self, queryloader, galleryloader, queryFliploader, galleryFliploader,
             ranks=[1, 2, 4, 5, 8, 10, 16, 20], eval_flip=False, re_ranking=False, savefig=False, test_distance='global'):

        self.model.eval()
        print('we use class evaluate.......')
        with torch.no_grad():
            qf, q_pids, q_camids, lqf = [], [], [], []
            for batch_idx, (imgs, pids, camids) in enumerate(queryloader):

                imgs = imgs.cuda()

                features, local_features = self.model(imgs)

                features = features.data.cpu()
                local_features = local_features.data.cpu()
                qf.append(features)
                lqf.append(local_features)
                q_pids.extend(pids)
                q_camids.extend(camids)
            qf = torch.cat(qf, 0)
            lqf = torch.cat(lqf, 0)
            q_pids = np.asarray(q_pids)
            q_camids = np.asarray(q_camids)

            print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

            gf, g_pids, g_camids, lgf = [], [], [], []
            for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
                if batch_idx%50 == 0:
                    print('gallery finished ',batch_idx*1.0/len(galleryloader))
                imgs = imgs.cuda()
                features, local_features = self.model(imgs)
                features = features.data.cpu()
                local_features = local_features.data.cpu()
                gf.append(features)
                lgf.append(local_features)
                g_pids.extend(pids)
                g_camids.extend(camids)
            gf = torch.cat(gf, 0)
            lgf = torch.cat(lgf, 0)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)

            print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        # feature normlization
        qf = 1. * qf / (torch.norm(qf, 2, dim=-1, keepdim=True).expand_as(qf) + 1e-12)
        gf = 1. * gf / (torch.norm(gf, 2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)
        print('normalization done.......')



        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

        if not test_distance == 'global':
            print("Only using global branch")
            lqf = lqf.permute(0, 2, 1)
            lgf = lgf.permute(0, 2, 1)
            local_distmat = low_memory_local_dist(lqf.numpy(), lgf.numpy(), aligned=True)
            if test_distance == 'local':
                print("Only using local branch")
                distmat = local_distmat
            if test_distance == 'global_local':
                print("Using global and local branches")
                distmat = local_distmat + distmat
        print("Computing CMC and mAP")
        cmc, mAP = self.evaluate_mat(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        return cmc[0]

