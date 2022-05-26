import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
L = 2 #how many layers?
nodeL = [] #how many nodes in each hidden layer?
train_x = None
train_y = None
train_verdict_y = []
test_x = None
test_y = None
test_verdict_y = []
y = None
A = None
error = 0
learning_rate = 0.001
wdict = {}
adict = {}
zdict = {}
ddict = {}
vdict = {}
np.set_printoptions(precision=2,
                       threshold=10000,
                       linewidth=150)

def size_weight_Init(reqlist):
    global L,nodeL,y,A
    L = len(reqlist)
    #0th node is input node
    #L-1 th node is output node
    np.random.seed(1)
    for i in range(len(reqlist)):
        n = reqlist[i]
        nodeL.append(n)
    #print(nodeL)
    print(L)
    input_size = nodeL[0]
    #print(input_size)
    A = np.random.randn(input_size,1) #taking random input
    #print(adict)
    output_size = nodeL[L-1]
    #y = np.zeros(shape=(output_size, 1))
    for i in range(0,L-1):
        row = nodeL[i+1] #number of nodes in next layer
        col = nodeL[i]  #number of nodes in this layer
        W = np.random.randn(row, col)#row = next layer, col = this layer
        V = np.random.randn(row, col)#row = next layer, col = this layer
        V.fill(0)
        Z = np.zeros(shape=(row, 1))
        A = np.zeros(shape=(row, 1))
        D = np.zeros(shape=(row, 1))
        #print("weight for Layer ", i+1)
        #print("W",W,"V",V,"Z",Z,"A",A,"D",D)
        #print(Z)
        wdict['Layer' + str(i)] = W #this layer
        vdict['Layer' + str(i)] = V #this layer
        zdict['Layer' + str(i+1)] = Z #next layer
        adict['Layer' + str(i+1)] = A #next layer
        ddict['Layer' + str(i+1)] = D #next layer
        #print('Layer' + str(i+1))
    #print(wdict)


def forward_Propagation():
    #print("Forward Propagation")
    for l in range(0, L-1):
        nodes1 = nodeL[l] #this layer
        nodes2 = nodeL[l+1] #next layer
        #print(nodes1,nodes2)
        #print('Layer'+str(l+1))
        W = wdict['Layer'+str(l)]#this layer
        a = adict['Layer'+str(l)]#this layer
        A = adict['Layer'+str(l+1)]#next layer
        Z = zdict['Layer'+str(l+1)]#next layer
        for n in range(0,nodes2):
            w = W[n]
            w = w.reshape(1,(len(w)))
            #print("Shape of w", np.shape(w), " shape of a",np.shape(a))
            zval = np.dot(w, a)
            #print(zval)
            #zval = zval[0][0]
            aval = 1 / (1 + np.exp(-zval))
            Z[n][0] = zval
            A[n][0] = aval
    #A = adict['Layer' + str(L-1)]
    #print('Output Layer: ',L-1,A)


def backward_Propagation():
    #print("Backpropagation")
    global y,L,nodeL,error
    A = adict['Layer' + str(L-1)]#final layer
    A_prime = adict['Layer' + str(L-1)]#final layer
    nodes = nodeL[L-1]
    D = ddict['Layer' + str(L-1)]
    for n in range(nodes):
        D[n][0] = (A[n][0] - y[n][0])* A_prime[n][0] *(1-A_prime[n][0])
    ddict['Layer' + str(L-1)] = D
    for l in range(L-1, 1, -1):
        nodes1 = nodeL[l]#this layer
        nodes2 = nodeL[l-1]#prev layer
        #print(nodes1,nodes2)
        #print('Layer' + str(l-1))
        W = wdict['Layer' + str(l-1)]
        W_T = [list(i) for i in zip(*W)]
        #W_T = np.transpose(W)
        A = adict['Layer' + str(l-1)]#prev layer
        d = ddict['Layer' + str(l-1)]#prev layer
        D = ddict['Layer' + str(l)]#this layer
        V = vdict['Layer' + str(l-1)]#prev layer
        for n in range(0, nodes2):
            #w = W_T[n]
            #x = np.dot(w, D)
            #print(x)
            #dval = x * A[n][0] * (1 - A[n][0])
            #print(d)
            d[n][0] = np.dot(W_T[n],D) * A[n][0] * (1 - A[n][0]) 
        #print('D Layer:',l-1,d)
        #print(np.shape(D),np.shape(A),np.shape(d))
        v = np.dot(D, np.transpose(A))
        V = np.add(V,v)
        vdict['Layer' + str(l-1)] = V
        #print('Layer' + str(l))
        #print('V Layer:',l-1,v)
        #print(D)
    D = ddict['Layer' + str(1)]
    A = adict['Layer' + str(0)]
    V = vdict['Layer' + str(0)]
    v = np.dot(D, np.transpose(A))
    #V = np.add(0, v)
    V = np.add(V, v)
    vdict['Layer' + str(0)] = V
    #print('Layer' + str(1))


def updateWeights(m):
    for l in range(0, L-1):
        W = wdict['Layer' + str(l)]
        V = vdict['Layer' + str(l)]
        coeff =  learning_rate
        #V = np.multiply((1/m), V)
        V = np.multiply(coeff, V)
        w = np.subtract(W, V)
        #V = np.subtract(V,V)
        wdict['Layer' + str(l)] = w
        #print(w)

def updateWeight():
    for l in range(0, L-1):
        W = wdict['Layer' + str(l)]
        V = vdict['Layer' + str(l)]
        w = np.subtract(W,V)
        wdict['Layer' + str(l)] = w

def train(train_sample_size):
    print("Train")
    global y
    epoch = 1000
    for e in range(epoch):
        for i in range(train_sample_size):
            t = train_x[i]
            A = np.reshape(t,(-1,1))
            adict['Layer' + str(0)] = A
            forward_Propagation()
            # store that verdict A(L) as train_verdict dictionary for each sample
            v = adict['Layer' + str(L-1)]
            #print(v)
            # res = adict['Layer' + str(L - 1)]
            # nodes = nodeL[L - 1]
            # for n in range(nodes):
            #     error += 0.5 * (res[n][0] - y[n][0]) ** 2
            train_verdict_y.append(v)
            t = train_y[i]
            y = np.reshape(t,(-1,1))
            backward_Propagation()
            #updateWeight()
        updateWeights(train_sample_size)
        train_verdict_y.clear()
    #save the current state of W's for all layers
    #print(error/(epoch * train_sample_size))


def test(test_sample_size):
    print("Test")
    error = 0
    for i in range(test_sample_size):
        t = test_x[i]
        yt = test_y[i]
        y = np.reshape(yt,(-1, 1))
        A = np.reshape(t, (-1, 1))
        adict['Layer' + str(0)] = A
        forward_Propagation()
        res = adict['Layer' + str(L - 1)]
        nodes = nodeL[L-1]
        error_per_sample = 0
        for n in range(nodes):
            if res[n][0] < 0.1:
                res[n][0] = 0
            error_per_sample += (res[n][0]-y[n][0])**2
        error += error_per_sample
        if error_per_sample > 0.1:
            print("Misclassify sample ",i)
        test_verdict_y.append(res)
        print(i, "Predicted" ,np.transpose(res), "Actual",yt)
    #error = (error **0.5)/test_sample_size
        #store that verdict A(L) as test_verdict dictionary for each sample
    error = error / 2
    print("Total squared error: ",error)

def onehotencoder(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print(integer_encoded)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)
    return onehot_encoded

def dataProcessing():
    global train_x,train_y,test_x,test_y
    scaler = MinMaxScaler()
    ###Path###
    train_data_path = 'D:\Study\Python Codes\\NNBackPropagation\Data\\Train1.csv'
    test_data_path = 'D:\Study\Python Codes\\NNBackPropagation\Data\\Test1.csv'
    ###Load Data###
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    ###Output###
    train_data_y = train_data['y']
    output_size = len(set(train_data_y.values))
    train_y = onehotencoder(train_data_y.values)
    train_data.drop(['y'], axis=1)
    test_data_y = test_data['y']
    test_y = onehotencoder(test_data_y.values)
    test_data.drop(['y'], axis=1)
    ###Input###
    train_data_x = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    train_x = train_data_x.values
    test_data_x = pd.DataFrame(scaler.fit_transform(test_data), columns=test_data.columns)
    test_x = test_data_x.values
    train_sample_size = len(train_data.axes[0])
    test_sample_size = len(test_data.axes[0])
    attribute_size = len(train_data.axes[1])
    return train_sample_size,test_sample_size,attribute_size,output_size


# def multiclassProcessing():
#     enc = OneHotEncoder()
#     cne = OneHotEncoder()
#     train_path = ""
#     test_path = ""
#     train_data = pd.read_csv(train_path)
#     test_data = pd.read_csv(test_path)
#     tray = enc.fit_transform(train_data.y.values.reshape(-1,1)).toarray()
#     tey = cne.fit_transform(test_data.y.values.reshape(-1,1)).toarray()
#     # t_y = train_data['y'].values
#     # train_x = train_data[train_data.drop(['y'],axis=1)].values
#     # t_y = test_data['y'].values
#     # test_x = test_data[test_data.drop(['y'],axis=1)].values


def main():
    #take an array as input
    Net_Structure = []
    train_sample_size, test_sample_size ,attribute_size ,output_size = dataProcessing()
    llength = int(input("The layers in NN excluding input and outputs: "))
    Net_Structure.append(attribute_size)
    for i in range(llength):
        x = int(input())
        Net_Structure.append(x)
    Net_Structure.append(output_size)
    #print(Net_Structure)
    size_weight_Init(Net_Structure)
    train(train_sample_size)
    #print(train_verdict_y)
    test(test_sample_size)


if __name__ == '__main__':
    main()