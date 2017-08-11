import numpy as np

# fr = open('/mnt/nas/CIKM/DATA/data_new/CIKM2017_train/train.txt', 'r')
# fr = open('testA.txt', 'r')
data = np.load('train_only_H1.npy')
print data.shape
data = data.reshape(10000,-1)
print data.shape
dataMat0 = []
dataMat1 = []
dataMat2 = []
dataMat3 = []
dataMat4 = []
dataMat5 = []
dataMat6 = []
dataMat7 = []
dataMat8 = []
dataMat9 = []
count = 0

for line in data:

    print count
    temp0 = np.array([])
    temp1 = np.array([])
    temp2 = np.array([])
    temp3 = np.array([])
    temp4 = np.array([])
    temp5 = np.array([])
    temp6 = np.array([])
    temp7 = np.array([])
    temp8 = np.array([])
    temp9 = np.array([])

    for i in range(15):
        if not (i in [11,12,13,14]):
            continue

        ######
        # here is the difference of generating diff feature or not.
        # cancal the following annotation will generate the diff feature.
        #####

        elif i in [11,14]: 
            mat = line[i*101*101:(i+1)*101*101].reshape(101,101)
        elif i in [12,13]:
            mat = line[(i+1)*101*101:(i+2)*101*101] - line[i*101*101:(i+1)*101*101]
            mat = mat.reshape(101,101) 

        ####
        ####   use this one will generate origin feature.

        # mat = line[i*101*101:(i+1)*101*101].reshape(101,101)
        
        ##########################

        mat0 = mat[25:76, 25:76].reshape(-1,1)
        temp0 = np.append(temp0, mat0)

        mat1 = (mat.T)[25:76, 25:76].reshape(-1,1)
        temp1 = np.append(temp1, mat1)  #transpose

        mat2 = np.rot90(mat,1)[25:76, 25:76].reshape(-1,1)
        temp2 = np.append(temp2, mat2)

        mat3 = np.rot90(mat,2)[25:76, 25:76].reshape(-1,1)
        temp3 = np.append(temp3, mat3)

        mat4 = np.rot90(mat,3)[25:76, 25:76].reshape(-1,1)
        temp4 = np.append(temp4, mat4)

        mat5 = np.flipud(mat)[25:76, 25:76].reshape(-1,1)
        temp5 = np.append(temp5, mat5)

        mat6 = np.fliplr(mat)[25:76, 25:76].reshape(-1,1)
        temp6 = np.append(temp6, mat6)

        mat7 = np.rot90(mat.T,1)[25:76, 25:76].reshape(-1,1)
        temp7 = np.append(temp7, mat7)

        mat8 = np.rot90(mat.T,2)[25:76, 25:76].reshape(-1,1)
        temp8 = np.append(temp8, mat8)

        mat9 = np.rot90(mat.T,3)[25:76, 25:76].reshape(-1,1)
        temp9 = np.append(temp9, mat9)

    dataMat0.append(temp0)
    del temp0
    dataMat1.append(temp1)
    del temp1
    dataMat2.append(temp2)
    del temp2
    dataMat3.append(temp3)
    del temp3
    dataMat4.append(temp4)
    del temp4
    dataMat5.append(temp5)
    del temp5
    dataMat6.append(temp6)
    del temp6
    dataMat7.append(temp7)
    del temp7
    dataMat8.append(temp8)
    del temp8
    dataMat9.append(temp9)
    del temp9

    count += 1
del data

data = np.append(dataMat0, dataMat1)
del dataMat0, dataMat1

data = np.append(data, dataMat2)
del dataMat2

data = np.append(data, dataMat3)
del dataMat3

data = np.append(data, dataMat4)
del dataMat4

data = np.append(data, dataMat5)
del dataMat5

data = np.append(data, dataMat6)
del dataMat6

# data = np.append(data, dataMat7)
del dataMat7

data = np.append(data, dataMat8)
del dataMat8

# data = np.append(data, dataMat9)
del dataMat9

data = data.reshape(80000,-1)
data = data.astype('uint8')
np.save('train_only_H1_aug_8_Last4_uint8_diff.npy', data)

label = np.load("label.npy")
label = np.append(label, label)
label = np.append(label, label)
label = np.append(label, label)
np.save('label_aug_8.npy', label)
del data
