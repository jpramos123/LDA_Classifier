import numpy as np
def matmul(mat1, mat2):
    """
        This function make a matrix multiplication
        or return an feedback if the multiplication
        isn't valid.
    """
    
    if len(mat1[0]) == len(mat2):

        n = len(mat1)
        p = len(mat2[0])
        m = len(mat1[0])
        
        mat3 = [[0 for i in range(p)] for j in range(n)]
        
        for i in range(n):
            for j in range(p):
                sum = 0
                for k in range(m):
                    sum = sum + mat1[i][k] * mat2[k][j]
                mat3[i][j] = sum
        return mat3

    else:
        print("Invalid multiplication")
        return -1

def find_determinant(mat):

    if len(mat) == 2:
        return (mat[0][0]*mat[1][1]) - (mat[0][1]*mat[1][0])
    elif len(mat) == 3:
        
        return ((mat[0][0]*mat[1][1]*mat[2][2]) 
              + (mat[0][1]*mat[1][2]*mat[2][0]) 
              + (mat[0][2]*mat[1][0]*mat[2][1]) 
              - (mat[0][2]*mat[1][1]*mat[2][0])
              - (mat[0][1]*mat[1][0]*mat[2][2])
              - (mat[0][0]*mat[1][2]*mat[2][1])
             )
    else:
        print("Not possible to calculate")
        return -1          
    
def transpose(mat):
    
    n_row = len(mat)
    n_col = len(mat[0])

    new_mat = [[0 for i in range(n_row)] for j in range(n_col)]

    for i in range(n_row):
        for j in range(n_col):
            new_mat[j][i] = mat[i][j]

    return new_mat


def invert_mat(mat):

    """
        This function invert a 2x2 or 3x3 matrix
        if high scaled matrix is passed it will return
        a feedback message
    """        

    if len(mat) == len(mat[0]):

        if len(mat) == 2:
            new_mat = [[0 for i in range(2)] for j in range(2)]
            determinant = find_determinant(mat)

            new_mat[0][0] = mat[1][1]
            new_mat[1][1] = mat[0][0] 
            new_mat[0][1] = (-mat[0][1])
            new_mat[1][0] = (-mat[1][0])

            for i in range(2):
                for j in range(2):
                    new_mat[i][j] = (1/determinant) * new_mat[i][j]
            
            return new_mat

        elif len(mat) == 3:
            new_mat = [[0 for i in range(3)] for j in range(3)]
            determinant = find_determinant(mat)

            new_mat[0][0] = +((mat[1][1] * mat[2][2]) - (mat[1][2] * mat[2][1]))
            new_mat[0][1] = -((mat[1][0] * mat[2][2]) - (mat[1][2] * mat[2][0]))
            new_mat[0][2] = +((mat[1][0] * mat[2][1]) - (mat[1][1] * mat[2][0]))
            new_mat[1][0] = -((mat[0][1] * mat[2][2]) - (mat[0][2] * mat[2][1]))
            new_mat[1][1] = +((mat[0][0] * mat[2][2]) - (mat[0][2] * mat[2][0]))
            new_mat[1][2] = -((mat[0][0] * mat[2][1]) - (mat[0][1] * mat[2][0]))
            new_mat[2][0] = +((mat[0][1] * mat[1][2]) - (mat[0][2] * mat[1][1]))
            new_mat[2][1] = -((mat[0][0] * mat[1][2]) - (mat[0][2] * mat[1][0]))
            new_mat[2][2] = +((mat[0][0] * mat[1][1]) - (mat[0][1] * mat[1][0]))

            for i in range(3):
                for j in range(3):
                    new_mat[i][j] = (1/determinant) * new_mat[i][j]
            return transpose(new_mat)
        

        else:
            print("Not possible to invert")
            return -1
    else:
        print("Not possible to invert")
        return -1            

def multiply_arr(arr, betha):
    new_arr = [v for v in arr]
    for i in range(len(arr)):
        new_arr[i] = float(new_arr[i]*betha)
    return new_arr

def sum_arr(arr, val_to_sum):
    new_arr = [v for v in arr]
    for i in range(len(arr)):
        new_arr[i] = float(new_arr[i] + val_to_sum)
    return new_arr

def buildData(data_frame):

    x = [[1] for i in range(data_frame.shape[0])]
    y = [[] for i in range(data_frame.shape[0])]
    for i in range(data_frame.shape[1]):
        for j in range(data_frame.shape[0]):
            if i < (data_frame.shape[1] - 1):
                # Getting features
                x[j].append(data_frame.iloc[j,i])
            else:
                y[j].append(data_frame.iloc[j,i])
    return [x,y]

def mean(arr):

    return sum(arr)/float(len(arr))

def mean_data_adjust(data):

    new_data = [[0 for i in range(len(data[0]))] for j in range(len(data))]

    for i in range(len(data)):
        for j in range(len(data[0])):
            
            new_data[i][j] = data[i][j] - mean(data[i])
    return new_data

def cov(arr1, arr2):

    mean1 = mean(arr1)
    mean2 = mean(arr2)
    n = len(arr1)

    cov_res = 0
    for i in range(len(arr1)):
        cov_res = float(cov_res + (arr1[i] - mean1) * (arr2[i] - mean2))
    cov_res = float(cov_res/(n-1))

    return cov_res

def cov_matrix(arr):

    cov_mat = [[0 for i in range(len(arr))] for j in range(len(arr))]

    for i in range(len(arr)):
        for j in range(len(arr)):
            cov_mat[i][j] = cov(arr[i],arr[j])
    return cov_mat

def calc_eigenvalues(cov_mat):

    import math

    a = 1
    b = ((-cov_mat[0][0]) + (-cov_mat[1][1]))
    c = (cov_mat[0][0] * cov_mat[1][1]) - (cov_mat[0][1] * cov_mat[1][0])

    x = (b**2)-(4*a*c)

    if x<0:
        print("Negative root")
        return None
    
    else:
        x = math.sqrt(x)
        x1 = (-b+x)/(2*a)
        x2 = (-b-x)/(2*a)
    
    return [x1,x2]

def calc_eingen(cov_mat):

    import numpy as np

    return [np.linalg.eigh(cov_mat)[1],np.linalg.eigh(cov_mat)[0]]

def makePCA(data):

    """This function is the pipeline for the PCA method"""

    final_data = transpose(data)
    final_data = mean_data_adjust(final_data)
    covariance_matrix = cov_matrix(final_data)
    eingen_arr = calc_eingen(covariance_matrix)

    #print(data)
    #print(final_data)
    #print(len([eingen_arr[0][1]][0]))
    #print(len(final_data))
   # print(matmul([eingen_arr[0][1]],final_data))
    return eingen_arr

def linear(dataset):

    beta = findBeta(dataset[0], dataset[1])
    #print("Beta Gerado: ", beta)
    n = len(beta)
    x = [i for i in range(-100,3000)]
    y = [0 for i in range(len(x))]

    
    for j in range(len(x)):
        for i in range(n-1):
            y[j] = y[j] + x[j]*beta[i+1][0]
        y[j] = y[j] + beta[0][0]
    
    return [x,y]

def quadratic(dataset):
    
    
    for v in dataset[0]:
        v.append(v[1]*v[1])

    beta = findBeta(dataset[0], dataset[1])

    print("Beta Gerado: ", beta)

    n = len(beta)
    x = [i for i in range(3000)]
    y = [0 for i in range(len(x))]

    for j in range(len(x)):
        y[j] = y[j] + x[j]*beta[1][0] + x[j]*x[j]*beta[2][0] + beta[0][0]

    return [x,y]

def findBeta(features, target):

    xt_x = matmul(transpose(features),features)
    xt_y = matmul(transpose(features),target)
    beta = matmul(invert_mat(xt_x),xt_y)
    return beta

def create_target_feature_dict(features, targets):
    targets_data_dict = {}
    for i in range(len(targets)):
        if targets[i] not in targets_data_dict.keys():
            targets_data_dict[targets[i]] = []
            targets_data_dict[targets[i]].append(features[i])
        else:
            targets_data_dict[targets[i]].append(features[i])

    for i in targets_data_dict.keys():
        targets_data_dict[i] = np.array(targets_data_dict[i])
    return targets_data_dict

def create_target_mean_dict(_dict):
    
    avg_dict = {}
    for k in _dict.keys():
        avg_dict[k] = np.sum(_dict[k], axis=0)/len(_dict[k])
    return avg_dict

def create_within_covariance_matrix(_avg_dict, _tgt_features_dict):
    cov_matrix_dict = {}
    for k in _avg_dict.keys():
        s = 0
        for i in range(len(_tgt_features_dict[k])):
            s = s + ((_tgt_features_dict[k][i] - _avg_dict[k])*np.transpose([(_tgt_features_dict[k][i] - _avg_dict[k])]))/(len(_tgt_features_dict[k])-1)
        cov_matrix_dict[k] = s
    return cov_matrix_dict

def find_within_class_scatter_matrix(_cov_matrix_dict):
    
    within_class_mtx = 0
    for k in _cov_matrix_dict.keys():
        within_class_mtx = within_class_mtx + _cov_matrix_dict[k]
    return (len(_cov_matrix_dict.keys())-1)*within_class_mtx

def find_grand_mean_vec(_features):
    return np.sum(_features, axis=0)/len(_features)

def create_between_covariance_matrix(_mean_by_target_dict, _gran_mean_vec):
    
    between_cov_mtx = {}
    for k in _mean_by_target_dict.keys():
        for i in range(len(_mean_by_target_dict[k])):
            between_cov_mtx[k] = (_mean_by_target_dict[k] - _gran_mean_vec)*np.transpose([_mean_by_target_dict[k] - _gran_mean_vec])
    return between_cov_mtx

def find_between_class_scatter_matrix(_between_cov_mtx_dict):
    
    between_class_mtx = 0
    for k in _between_cov_mtx_dict.keys():
        between_class_mtx = between_class_mtx + _between_cov_mtx_dict[k]
        
    between_class_mtx = len(_between_cov_mtx_dict) * between_class_mtx
    return between_class_mtx
    



if __name__ == "__main__":
    

    mat2 = [[2.5,  2.4],
            [0.5, 0.7],
            [2.2,2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2, 1.6],
            [1, 1.1],
            [1.5, 1.6],
            [1.1, 0.9]
            ]


    
"""
    print("** Books Attend Grade **")

    data = pd.read_csv("Books_attend_grade.txt", delim_whitespace=True)
    data = data.values.tolist()
    
    ein_arr = makePCA(data)
    print(ein_arr)

    print("** Alpswater **")

    data = pd.read_csv("alpswater.txt", delim_whitespace=True)
    data = data.values.tolist()
    
    ein_arr = makePCA(data)
    print(ein_arr)
"""