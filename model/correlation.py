r""" Provides functions that builds/manipulates correlation tensors """
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids, s_mask):
        eps = 1e-5
        corrs = []
        sups=[]
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            queryShape = query_feat.shape#b,c,h,w
            corrI=[]
            realSupI=[]
            mask = F.interpolate(s_mask, size=(queryShape[-2], queryShape[-1]),  
                                     mode='bilinear', align_corners=True)
            for j in range(len(support_feat)):#b
                mask_use = mask[j]
                queryIJ = query_feat[j].flatten(start_dim = 1) # c, hw
                queryIJNorm = queryIJ/(queryIJ.norm(dim=0, p=2, keepdim=True) + eps)
                supIJ = support_feat[j].flatten(start_dim = 1).transpose(-2, -1).contiguous() # hw x c
                supIJNorm=supIJ/(supIJ.norm(dim=1, p=2, keepdim=True) + eps)
                corr = supIJNorm.matmul(queryIJNorm) # hw x hw 
                corr = corr.permute(1, 0) # hw x hw 
                corr= F.softmax(corr, dim=-1) # softmax
                corr = corr.matmul(mask_use.flatten(start_dim = 1).transpose(-2, -1)).squeeze(-1)
                similarity = (corr - corr.min(0)[0].unsqueeze(0)) / (
                    corr.max(0)[0].unsqueeze(0) - corr.min(0)[0].unsqueeze(0) + eps)
                similarity = similarity.view(1, 1, -1)
                corrI.append(similarity)
            corrI=torch.cat(corrI,dim=0)#bsz x 1 xhw
            corrI=corrI.reshape((corrI.shape[0],corrI.shape[1],queryShape[-2],queryShape[-1]))#b,1,h,w
            corrs.append(corrI)#n,b,1,h,w

        corr_l4 = torch.cat(corrs[-stack_ids[0]:],dim=1).contiguous()#b,n,h,w
        corr_l3 = torch.cat(corrs[-stack_ids[1]:-stack_ids[0]],dim=1).contiguous()
        corr_l2 = torch.cat(corrs[-stack_ids[2]:-stack_ids[1]],dim=1).contiguous()

        return [corr_l4, corr_l3, corr_l2] #,[sup_l4,sup_l3,sup_l2]



    @classmethod
    def multilayer_correlation_hsnet(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            # support_feat_norm = torch.norm(support_feat, dim=1, p=2, keepdim=True)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            # query_feat_norm = torch.norm(query_feat, dim=1, p=2, keepdim=True)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)


            corr = torch.bmm(query_feat.transpose(1, 2), support_feat)
            corr = corr.clamp(min=0)
            corr = corr.mean(dim=2,keepdim=True).squeeze(2)
            corr = corr.view(bsz, hb, wb)
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]
    

    @classmethod
    def multilayer_correlation_query(cls, query_feats, support_feats, stack_ids, q_mask):
        eps = 1e-5
        corrs = []
        sups=[]
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            queryShape = query_feat.shape#b,c,h,w
            corrI=[]
            realSupI=[]
            mask = F.interpolate(q_mask, size=(queryShape[-2], queryShape[-1]),  
                                     mode='bilinear', align_corners=True)
            for j in range(len(support_feat)):#b
                mask_use = mask[j]
                queryIJ = query_feat[j].flatten(start_dim = 1) # c, hw
                supIJ = support_feat[j].flatten(start_dim = 1).transpose(-2, -1).contiguous() # hw x c
                corr = supIJ.matmul(queryIJ) # hw x hw 行是support
                corr = corr.permute(1, 0) # hw x hw 行是query
                corr= F.softmax(corr, dim=-1) # softmax
                corr = corr.matmul(mask_use.flatten(start_dim = 1).transpose(-2, -1)).squeeze(-1)
                similarity = (corr - corr.min(0)[0].unsqueeze(0)) / (
                    corr.max(0)[0].unsqueeze(0) - corr.min(0)[0].unsqueeze(0) + eps)
                similarity = similarity.view(1, 1, -1)
                corrI.append(similarity)
               
            corrI=torch.cat(corrI,dim=0)#bsz x 1 xhw
            corrI=corrI.reshape((corrI.shape[0],corrI.shape[1],queryShape[-2],queryShape[-1]))#b,1,h,w
            corrs.append(corrI)#n,b,1,h,w

        corr_l4 = torch.cat(corrs[-stack_ids[0]:],dim=1).contiguous()#b,n,h,w
        corr_l3 = torch.cat(corrs[-stack_ids[1]:-stack_ids[0]],dim=1).contiguous()
        corr_l2 = torch.cat(corrs[-stack_ids[2]:-stack_ids[1]],dim=1).contiguous()

      
        return [corr_l4, corr_l3, corr_l2] #,[sup_l4,sup_l3,sup_l2]