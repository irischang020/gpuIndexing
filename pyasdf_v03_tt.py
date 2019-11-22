import time
import math
import numpy as np
import torch


#reflections = np.load("../data/refls.npy")
#reflections = torch.from_numpy(reflections)
#reflections = reflections.to(device)
#lattice_off_max = 1./1000      # in Ainv
#predict_off_max = 1./500       # in Ainv
#lattice_min = 10               # in A
#lattice_max = 500.             # thrice the maximum expected axis length
#volume_min = 1000.             # in A^3
#volume_max = 100000000.        # in A^3
#N_reflections = len(reflections)
#N_bragg_triplets_max = 10000   # maximum number of triplets

torch.manual_seed(0)

def sample_bragg_triplets(N_reflections, N_bragg_triplets_max):
    tic = time.time()
    N_bragg_triplets = min( N_bragg_triplets_max, N_reflections*(N_reflections-1)*(N_reflections-2)/6 )
    random_bragg_triplets_base10 = torch.randperm(N_reflections**3)[:int(N_bragg_triplets)]
    bragg_triplets = torch.zeros([3,N_bragg_triplets],dtype=torch.int64)
    bragg_triplets[2] = random_bragg_triplets_base10 % N_reflections
    random_bragg_triplets_base10 //= N_reflections
    bragg_triplets[1] = random_bragg_triplets_base10 % N_reflections
    random_bragg_triplets_base10 //= N_reflections
    bragg_triplets[0] = random_bragg_triplets_base10
    print("## Measure ## Inside sample_bragg_triplets: ", time.time() - tic)
    return bragg_triplets


def sample_normal_vectors(reflections, bragg_triplets):
    """
    This function returns possible normal vectors of bragg triplets
    """
    tic = time.time()
    vectors = torch.cross(reflections[bragg_triplets[1]] - reflections[bragg_triplets[0]], reflections[bragg_triplets[2]] - reflections[bragg_triplets[0]]).T
    vectors_length = torch.norm(vectors, dim=0)
    index = torch.where(vectors_length>0.0001)
    normal_vectors = vectors[:,index[0]] / vectors_length[index]
    print("## Measure ## Inside sample_normal_vectors: ", time.time() - tic)
    return normal_vectors * torch.sign(normal_vectors[0])


def normal_vectors_filter(normal_vectors):
    tic = time.time()
    x = ((normal_vectors + 1) * 10).floor().type(torch.int16)
    vectors_feature = x[0]*20**2 + x[1]*20 + x[2]
    vectors_feature = vectors_feature.cpu().numpy()
    _,index = np.unique(vectors_feature,return_index=True)
    new_normal_vectors = normal_vectors[:,index]
    print("## Measure ## Inside normal_vectors_filter: ", time.time() - tic)
    return new_normal_vectors


def make_projection(reflections, normal_vectors):
    tic = time.time()
    projections = torch.mm(reflections,normal_vectors)
    print("## Measure ## Inside make_projection: ", time.time() - tic)
    return projections


def projections_find_lattice(projections,normal_vectors,lattice_min,lattice_max,lattice_off_max):
    tic = time.time()
    holder_length = 256
    N_reflections, N_normal_vectors = projections.shape
    rowmin, rowmax = torch.min(projections,0)[0].cuda(), torch.max(projections,0)[0].cuda()
    allmin,allmax = min(rowmin), max(rowmax)
    projections_holder = (projections - rowmin) / (rowmax - rowmin) * (holder_length-1)
    ps, ns = torch.meshgrid(projections_holder.view(-1).type(torch.DoubleTensor), torch.arange(holder_length).type(torch.DoubleTensor))
    ps = ps.view((N_reflections, N_normal_vectors, holder_length))
    ns = ns.view((N_reflections, N_normal_vectors, holder_length))
    dft = (torch.sqrt(sum(torch.cos(math.pi*ps*ns/holder_length))**2+sum(torch.sin(math.pi*ps*ns/holder_length))**2))
    i_max = int(lattice_max * (allmax - allmin))
    dft_freq = (torch.argmax(dft[:, 4:i_max],dim=1) + 4).cuda()
    dft_peak = torch.max(dft[:, 4:i_max],1)
    projections_lattice = (dft_freq / 2. / (rowmax - rowmin))
    num_fitted_bragg = torch.sum(torch.abs(projections - torch.round(projections*projections_lattice) / projections_lattice) < lattice_off_max, dim=0)
    index = torch.where((dft_peak[0].cuda()>N_reflections/2.)&(num_fitted_bragg>N_reflections/3.)&(projections_lattice>lattice_min)&(projections_lattice<lattice_max))[0]
    print("## Measure ## Inside projections_find_lattice: ", time.time() - tic)
    return projections[:,index], normal_vectors[:,index], projections_lattice[index]


def refine_projections_lattice(projections, projections_lattice, lattice_off_max):
    tic = time.time()
    lattice_old = projections_lattice.clone()
    for _ in range(20):
        projections_fitted_index = (torch.abs(projections - torch.round(projections*lattice_old) / lattice_old) < lattice_off_max)
        projections_fitted = projections * projections_fitted_index
        fitted_indices = torch.round(projections_fitted * lattice_old)
        lattice_new = torch.sum(fitted_indices * projections_fitted, dim=0) / torch.sum(projections_fitted * projections_fitted, dim=0)
        if torch.max(torch.abs(lattice_old - lattice_new)) < 1e-2:
            break
        lattice_old = lattice_new.clone()
    print("## Measure ## Inside refine_projections_lattice: ", time.time() - tic)
    return lattice_new, torch.max(torch.sum(projections_fitted_index,dim=0))


def build_lattice_vector(normal_vectors, projections_lattice):
    tic = time.time()
    lattice_vector = normal_vectors * projections_lattice
    index = torch.argsort(projections_lattice)
    print("## Measure ## Inside build_lattice_vector: ", time.time() - tic)
    return lattice_vector[:,index]


def calculate_Amatrix(invAmatrix_block):
    return torch.inverse(invAmatrix_block)


def calculate_volume(invAmatrix_block):
    volume_block = torch.sum(torch.cross(invAmatrix_block[:,0],invAmatrix_block[:,1],axis=1)*invAmatrix_block[:,2],dim=1)
    return volume_block


def build_invAmatrix_block(lattice_vector,reflections,predict_off_max):
    tic = time.time()
    N_reflections = len(reflections)
    N_lattice_vector = lattice_vector.shape[1]
    i,j,k = torch.meshgrid(torch.LongTensor(range(N_lattice_vector)),torch.LongTensor(range(N_lattice_vector)),torch.LongTensor(range(N_lattice_vector)))
    i = i.reshape(-1)
    j = j.reshape(-1)
    k = k.reshape(-1)

    index = torch.where((i<j)&(j<k))
    i = i[index]
    j = j[index]
    k = k[index]

    ijk=i+j+k
    index = torch.argsort(ijk)
    i = i[index]
    j = j[index]
    k = k[index]

    invAmatrix_block = torch.tensor([lattice_vector[:,i].cpu().numpy(),lattice_vector[:,j].cpu().numpy(),lattice_vector[:,k].cpu().numpy()]).permute(2,0,1) # In shape NA * 3 * 3 row
    volume_block = calculate_volume(invAmatrix_block)    # in shape (NA,) 
    HKL_block = torch.matmul(invAmatrix_block.cuda(),reflections.T.cuda())
    predict_off = torch.matmul(calculate_Amatrix(invAmatrix_block).cuda(),torch.round(HKL_block).cuda()) - reflections.T.cuda()
    predict_off = predict_off.cpu().numpy()
    num_predict_fit = np.sum(np.sum(predict_off**2, axis=1) < predict_off_max**2, axis=1)
    volume_block = volume_block.cpu().numpy()
    invAmatrix_block = invAmatrix_block.cpu().numpy()
    index = np.where((volume_block**2 > np.prod(np.sum(invAmatrix_block**2,axis=2),axis=1)/10000.) & (num_predict_fit > N_reflections/6.))
    print("## Measure ## Inside build_Amatrix_block: ", time.time() - tic)
    return torch.from_numpy(invAmatrix_block[index])


def refine_invAmatrix_block(invAmatrix_block,reflections,predict_off_max):
    ## invAmatrix_block: in shape (NA * 3 * 3 row)
    ## reflections: in shape (NR * 3)
    ## predict_off_max: integer, the maximum allowed shift of a Bragg peak
    ## num_fitted_bragg_max: integer, the maximum number of fitted Bragg peaks of a single t vector
    tic = time.time()
    N_invAmatrix_block = len(invAmatrix_block)
    N_reflections = len(reflections)
    HKL_block = invAmatrix_block.matmul(reflections.T)      # in shape (NA * 3 * NR row) 
    predict_off = reflections.T - torch.matmul(calculate_Amatrix(invAmatrix_block),torch.round(HKL_block))  # in shape (NA * 3 * NR row)
    predict_off_fit_index = torch.from_numpy(np.sum(predict_off.cpu().numpy()**2, axis=1)) < predict_off_max**2   # in shape (NA * NR)

    for _ in range(20):
        X = torch.from_numpy((np.ones((3,N_invAmatrix_block,N_reflections)) * predict_off_fit_index.cpu().numpy()).transpose(1,0,2) * (reflections.T).cpu().numpy())
        y = torch.from_numpy( ((torch.round(HKL_block).cpu().numpy()).transpose(1,0,2) * predict_off_fit_index.cpu().numpy() ).transpose(1,0,2))  # in shape (NA * 3 * NR)
        X_permute_021 = torch.from_numpy(X.cpu().numpy().transpose(0,2,1))
        y_permute_021 = torch.from_numpy(y.cpu().numpy().transpose(0,2,1))
        invAmatrix_block_new = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X, X_permute_021)), X), y_permute_021).permute(0,2,1)
        HKL_block_new = torch.matmul(invAmatrix_block_new.cuda(),reflections.T.cuda())
        predict_off = reflections.T - torch.matmul(calculate_Amatrix(invAmatrix_block_new).cuda(),torch.round(HKL_block).cuda())
        predict_off_fit_index = torch.sum(predict_off**2, dim=1) < predict_off_max**2   # in shape (NA * NR)
        if torch.max(torch.abs(HKL_block_new - HKL_block))<1e-3:
            break
        HKL_block = HKL_block_new.clone()
    num_predict_fit = torch.sum(predict_off_fit_index,dim=1)
    print("## Measure ## Inside refine_invAmatrix_block: ", time.time() - tic)
    return invAmatrix_block_new, num_predict_fit


def find_best_invAmatrix(invAmatrix_block,num_predict_fit,num_fitted_bragg_max):
    min_volume_index = 1e9
    index = torch.where(num_predict_fit>=num_fitted_bragg_max)
    if len(index[0]) > 0:
        min_volume_index = index[0][0]
    max_fitted_index = torch.argmax(num_predict_fit)
    return invAmatrix_block[min(min_volume_index,max_fitted_index)]


def get_angle(x,y):
    return torch.acos(torch.dot(x,y)/torch.norm(x)/torch.norm(y)) * 180. / math.pi


def get_lattice(mat):
    x =  torch.tensor([torch.norm(mat[0]),torch.norm(mat[1]),torch.norm(mat[2]),get_angle(mat[0],mat[1]),get_angle(mat[0],mat[2]),get_angle(mat[2],mat[1])])
    n_digits = 2
    return (x*10*n_digits).round()/(10*n_digits)


def reduce_unit_cell(invAmatrix):
    """
    This function was put to the end after selecting the best unit cell, it's not vectorized though.
    """
    changed = 1
    n = 0
    while changed:
        n += 1
        changed = 0

        v1 = invAmatrix[0].clone()
        v2 = invAmatrix[1].clone()
        v3 = invAmatrix[2].clone()

        a = torch.norm(v1)
        b = torch.norm(v2)
        c = torch.norm(v3)

        gamma = get_angle(v1,v2)
        alpha = get_angle(v2,v3)
        beta  = get_angle(v3,v1)

        if changed == 0:
            if gamma < 90:
                v2 = -v2
                gamma = 180-gamma
                alpha = 180-alpha
            v2 = v2+v1
            bb = torch.norm(v2)
            if bb < b:
                b = bb
                if a < b:
                    invAmatrix[1] = v2.clone()
                else:
                    invAmatrix[0] = v2.clone()
                    invAmatrix[1] = v1.clone()
                changed = 1

        if changed == 0:
            if beta < 90:
                v3 = -v3
                beta = 180-beta
                alpha = 180-alpha
            v3 = v3+v1
            cc = torch.norm(v3)
            if cc < c:
                c = cc
                if b < c:
                    invAmatrix[2] = v3.clone()
                elif a < c:
                    invAmatrix[1] = v3.clone()
                    invAmatrix[2] = v2.clone()
                else:
                    invAmatrix[0] = v3.clone()
                    invAmatrix[1] = v1.clone()
                    invAmatrix[2] = v2.clone()
                changed = 1

        if changed == 0:
            if alpha < 90:
                v3 = -v3
                beta = 180 - beta
                alpha = 180 - alpha

            v3 = v3+v2
            cc = torch.norm(v3)
            if cc < c:
                c = cc
                if b < c:
                    invAmatrix[2] = v3.clone()
                elif a < c:
                    invAmatrix[1] = v3.clone()
                    invAmatrix[2] = v2.clone()
                else:
                    invAmatrix[0] = v3.clone()
                    invAmatrix[1] = v1.clone()
                    invAmatrix[2] = v2.clone()
                changed = 1

        if n > 30:
            changed = 0
    return invAmatrix


def run_index(reflections):
    """
    **reflections saves Bragg peak positions in the reciprocal space, it must be provided as Nf * 3 matrix
    **givencell is provided unit cell informaton in CrystFEL format
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lattice_off_max = 1./1000        # in Ainv
    predict_off_max = 1./500          # in Ainv
    lattice_min = 10.                 # in A
    lattice_max = 500.                #  thrice the maximum expected axis length
    volume_min = 1000.                ## in A^3
    volume_max = 100000000.           ## in A^3
    N_reflections = len(reflections)
    N_bragg_triplets_max = 10000      # maximum number of triplets

    tic = time.time()
    bragg_triplets = sample_bragg_triplets(N_reflections, N_bragg_triplets_max).to(device)
    normal_vectors = sample_normal_vectors(reflections, bragg_triplets).to(device)
    normal_vectors = normal_vectors_filter(normal_vectors).to(device)
    projections = make_projection(reflections, normal_vectors).to(device)
    projections, normal_vectors, projections_lattice = projections_find_lattice(projections,normal_vectors,lattice_min,lattice_max,lattice_off_max)
    #projections = projections_find_lattice(projections,normal_vectors,lattice_min,lattice_max,lattice_off_max)[0].to(device)
    #normal_vectors = projections_find_lattice(projections,normal_vectors,lattice_min,lattice_max,lattice_off_max)[1].to(device)
    #projections_lattice = projections_find_lattice(projections,normal_vectors,lattice_min,lattice_max,lattice_off_max)[2].to(device)
    projections_lattice, num_fitted_bragg_max = refine_projections_lattice(projections, projections_lattice, lattice_off_max)
    lattice_vectors = build_lattice_vector(normal_vectors, projections_lattice).to(device)
    invAmatrix_block = build_invAmatrix_block(lattice_vectors,reflections,predict_off_max).to(device)
    invAmatrix_block, num_predict_fit = refine_invAmatrix_block(invAmatrix_block,reflections,predict_off_max)
    best_invAmatrix = find_best_invAmatrix(invAmatrix_block,num_predict_fit,num_fitted_bragg_max).to(device)
    reduced_invAmatrix = reduce_unit_cell(best_invAmatrix).to(device)
    print("## Measure ## Total time, ", time.time() - tic)
    return {"reciprocal":calculate_Amatrix(reduced_invAmatrix).to(device),"cell":get_lattice(best_invAmatrix).to(device)}
