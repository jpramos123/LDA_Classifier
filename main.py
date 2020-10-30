from utils import *
from sklearn import datasets
import numpy as np
from collections import defaultdict

def print_classes(tp, _dict):
    for k in _dict.keys():
        print("{} Class {}\n".format(tp,k), _dict[k])
        print("\n")
if __name__ == "__main__":

    data = datasets.load_iris()

    features = data.data
    targets = data.target
    


    tgt_ftr_dict = create_target_feature_dict(features, targets)

    mean_by_target = create_target_mean_dict(tgt_ftr_dict)

    print_classes("Mean", mean_by_target)

    cov_matrix_by_tgt = create_within_covariance_matrix(mean_by_target, tgt_ftr_dict)
    
    print_classes("Covariance Matrix", cov_matrix_by_tgt)

    within_class_matrix = find_within_class_scatter_matrix(cov_matrix_by_tgt)

    print("Within Class Matrix\n", within_class_matrix, "\n\n")

    gran_mean_vec = find_grand_mean_vec(features)

    between_cov_mtx = create_between_covariance_matrix(mean_by_target, gran_mean_vec)

    between_class_matrix = find_between_class_scatter_matrix(between_cov_mtx)

    print("Between Class Matrix\n", between_class_matrix, "\n\n")

    sw_1_sb = np.matmul(np.linalg.inv(within_class_matrix),between_class_matrix)

    print("Sw^1 * Sb\n", sw_1_sb, "\n\n")

    eigs = np.linalg.eig(sw_1_sb)
 
    print("Eingen Values \n", eigs[0], "\n\n")
    print("Eingen Vectors \n", eigs[1], "\n\n")