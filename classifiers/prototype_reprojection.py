import torch
import numpy as np

# ----------------------------------
# prototype reprojection module
# ----------------------------------
def get_reproj_dist(query, support, beta):
    # query: batch_size, d
    # support: way, 1 , d
    lam = support.size(1) / support.size(2)
    rho = beta.exp().cuda()
    st = support.permute(0, 2, 1)  # way, d, 1
    
    # correspond to Equation 6 in the paper
    sst = support.matmul(st) # support * st
    sst_plus_ri = sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)
    sst_plus_ri_np = sst_plus_ri.detach().cpu().numpy()
    sst_plus_ri_inv_np = np.linalg.inv(sst_plus_ri_np) 
    sst_plus_ri_inv = torch.tensor(sst_plus_ri_inv_np).cuda()
    w = query.matmul(st.matmul(sst_plus_ri_inv))  # way, d, d
    Q_bar = w.matmul(support).mul(rho)  # way, batch_size, d

    dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0).neg()
    return dist