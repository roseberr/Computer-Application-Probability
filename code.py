import csv
import random
import numpy as np#exp 사용하기위해서 시그마 코브 행렬 값 할당 5장에서 determinant사용
#transpose하는데 사용

import math  # sqrt와 pi사용


with open('all.txt', 'r') as f:
    reader = csv.reader(f)
    flower_list = list(reader)
    random.Random(4).shuffle(flower_list)

kfold_training = []
kfold_test = []
# kfold training 과 test를 나누기 위한 단계
for i in range(0, 120):
    kfold_training.append(flower_list[i])

for i in range(120, 150):
    kfold_test.append(flower_list[i])

# [0]:s_length [1]:s_width [2]:p_length [3]p_width [4]:class
# class1 "Iris-setosa"   class2 "Iris-versicolor" class3 "Iris-virginica"

def return_max_min_value(parameter):#해당 파라미터의 data의 최대값과 최솟값 리턴 함수
    mincount = 9999
    maxcount = 0

    for i in range(0, 120):
        if (float(kfold_training[i][parameter]) < mincount):
            mincount = float(kfold_training[i][parameter])
        if (float(kfold_training[i][parameter]) > maxcount):
            maxcount = float(kfold_training[i][parameter])
    return maxcount, mincount


for i in range(0, 120): # float로 변환
    for j in range(0, 4):
        kfold_training[i][j] = float(kfold_training[i][j])

for i in range(0, 30):  # float로 변환
    for j in range(0, 4):
        kfold_test[i][j] = float(kfold_test[i][j])

def return_maxcount_class(L, R, parameter):
    str = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   # 첫번쨰 파라미터:위치 두번쨰 파라미터:class


    for i in range(0, 120):
        if kfold_training[i][parameter] < L:
            for j in range(0, 3):
                if kfold_training[i][4] == str[j]:
                    list[0][j] += 1
        elif kfold_training[i][parameter] > R:
            for j in range(0, 3):
                if kfold_training[i][4] == str[j]:
                    list[2][j] += 1
        else:
            for j in range(0, 3):
                if kfold_training[i][4] == str[j]:
                    list[1][j] += 1
    return_str = []
    for i in range(0, 3):
        if list[i][0] > list[i][1] and list[i][0] > list[i][2]:
            return_str.append(str[0])
        elif list[i][1] > list[i][0] and list[i][1] > list[i][2]:
            return_str.append(str[1])
        else:
            return_str.append(str[2])
    return return_str


def order_str(parameter):    # B-1에서 class의 순서 정하기

    mean_str = [[0, "Iris-setosa"], [0, "Iris-versicolor"], [0, "Iris-virginica"]]
    count_str = [0, 0, 0]

    for i in range(0, 120):
        for k in range(0, 3):
            if (kfold_training[i][4] == mean_str[k][1]):
                mean_str[k][0] += kfold_training[i][parameter]
                count_str[k] += 1

    mean_str[0][0] = mean_str[0][0] / count_str[0]
    mean_str[1][0] /= count_str[1]
    mean_str[2][0] /= count_str[2]
    mean_str.sort()
    return mean_str[0][1], mean_str[1][1], mean_str[2][1]



#B_1
print("B-1--------------------------------------------------------------------------------")

decision_LRvalue=[]
feature_str=['setal_length','setal_width','petal_length','petal_width']

for parameter in range(0,4):  #4개의 성분 전체돌리기
   maxcount,mincount=return_max_min_value(parameter)    #값중 제일 작은 값부터 제일 큰값까지 돌리기
   error_min=0
   str=order_str(parameter)
   for i in range(int(mincount*10),int(maxcount*10)):#왼쪽선
      for j in range(i+1,int(maxcount*10)):   #오른쪽선
         count=0    
         for k in range(0,120):# 전체돌기         

            if kfold_training[k][4]==str[0] and (i/10)>kfold_training[k][parameter]:
                  count+=1         
            if kfold_training[k][4]==str[1] and ((i/10)< kfold_training[k][parameter] and (j/10)>kfold_training[k][parameter]):
                  count+=1

            if kfold_training[k][4]==str[2] and (j/10)<kfold_training[k][parameter]:
                  count+=1

         if error_min<count:
            error_min=count
            confidence_L=i/10
            confidence_R=j/10
   s=feature_str[parameter]
   print("feature가 "+s+"일떄:-------------------------")

   print("error_min의 값은")
   print(error_min)
  
   print("Left decision value?")
   print(confidence_L)
   print("Right decision value?")
   print(confidence_R)
   decision_LRvalue.append([confidence_L,confidence_R])
   print()


# B_2

print("\n\n\n")
print("B-2--------------------------------------------------------------------------------")


def sigmoid(z):
    return 1 / (1 + math.exp(z * (-1)))


def H_func(w0, w1, x, y):
    return sigmoid(1 + w0 * x + w1 * y)


def c(w0, w1, x, y, z):
    return (-1) * (z * math.log(H_func(w0, w1, x, y)) - (z - 1) * math.log(1 - H_func(w0, w1, x, y)))


str_parameter = ["s_length", "s_width", "p_length", "p_width"]
str = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

for parameter1 in range(0, 4):  # 가로 전체돌리기
    for parameter2 in range(parameter1 + 1, 4):  # 세로 전체돌리기
        for k in range(0, 3):   # class 별로 돌리기
            W0 = 0.5
            W1 = 0.5
            a = 0.1
            for i in range(0, 10000):  # 10000번의 경사하강법
                sumx = 0
                sumy = 0

                for j in range(0, 120):    # 전체돌리기
                    if (str[k] == kfold_training[j][4]):
                        z = 1
                    else:
                        z = 0

                    # hx=H_func(W0,W1,kfold_training[j][parameter1],kfold_training[j][parameter2])
                    if (H_func(W0, W1, kfold_training[j][parameter1], kfold_training[j][parameter2]) >= 0.5):
                        hx = 1
                    else:
                        hx = 0

                    sumx += (hx - z) * kfold_training[j][parameter1]
                    sumy += (hx - z) * kfold_training[j][parameter2]

                sumx /= 120
                sumy /= 120
                #print(sumx)
                #print(sumy)
                W0 -= sumx * a
                W1 -= sumy * a
                if sumx==0 and sumy==0: break
            print("x축은 %s이고 y축은 %s 이다" % (str_parameter[parameter1], str_parameter[parameter2]))
            print("class  "+str[k]+"  판별")
            #   print("판별직선에서 W0: %f   W1: %f" %(W0,W1))
            print("W0: {0} W1:{1}".format(W0,W1))
            #print("판별직선에서 : %fx+%fy+1=0" % (W0, W1))
            print("판별직선에서 y=%fx+%f" % (-W0/W1,-1/W1))
            print("\n\n")




# B_3


print("\n\n\n")
print("B-3--------------------------------------------------------------------------------")


str=["Iris-setosa","Iris-versicolor","Iris-virginica"]

#for i in range (0,4):
#    print("%f %f" %(decision_LRvalue[i][0],decision_LRvalue[i][1]))


def set_Matrix(parameter):
    
    str=["Iris-setosa","Iris-versicolor","Iris-virginica"]
    Matrix = [[0] * 4 for t in range(100)] # 확률 계산하기 위한 배열 0 에는 class 0 3에는 총합
    for k in range(0,120):
        for j in range(0, 3):  # 3개의 클래스에 대해 Matrix 값 계산하기 위해서 돌리기
            if (kfold_training[k][4] == str[j]):
                Matrix[int(kfold_training[k][parameter] * 10)][j] += 1  # 두번재 index=class 첫번재 index=값
                Matrix[int(kfold_training[k][parameter] * 10)][3] += 1
                break
            else:
                continue
    return Matrix

def set_likelihood_p3(test, cn):
 
    return_val=1
    for parameter in range(0,4):#feature 4개돌리기
        orderstr = order_str(parameter)   # class 순서 정하기
        Matrix=set_Matrix(parameter)
        if Matrix[int(kfold_test[test][parameter] * 10)][cn]==0:
            if kfold_test[test][parameter] < decision_LRvalue[parameter][0]:
                if orderstr[0] == kfold_test[test][4]:
                    k = 1
                else:
                    k = 0.01
            elif kfold_test[test][parameter] >= decision_LRvalue[parameter][1]:
                if orderstr[2] == kfold_test[test][4]:
                    k= 1
                else:
                    k = 0.01
            else:
                if orderstr[1] == kfold_test[test][4]:
                    k = 1
                else:
                    k =0.01    
            return_val*=k
        else:
            return_val*=Matrix[int(kfold_test[test][parameter] * 10)][cn]
    return return_val

def set_prior():
    for i in range(0,30):
        for j in range(0,3):
            if (kfold_test[i][4] == str[j]):
                list_prior[j]+=1
                break
            else:
                continue
    for j in range (0,3):
        list_prior[j]/=30
    return list_prior




True_count=0; False_count=0
list_prior = [0, 0, 0, 0]
list_prior=set_prior()
list_Fp=[0,0,0]
list_Fn=[0,0,0]
list_Tn=[0,0,0]
for i in range(0,30):#test set 돌리기
    list_posterior = [1, 1, 1]

    #list_likelihood = [0, 0, 0, 0]

    for j in range(0, 3):

        list_posterior[j]= set_likelihood_p3(i, j) / list_prior[j]

    if list_posterior[0]>list_posterior[1] and list_posterior[0]>list_posterior[2]:
        if str[0]==kfold_test[i][4]:
            True_count+=1
            
        else:
            False_count+=1
            list_Fp[0]+=1
            if kfold_test[i][4]==str[1]:
                list_Fn[1]+=1
            elif kfold_test[i][4]==str[2]:
                list_Fn[2]+=1

    elif (list_posterior[1]>list_posterior[2] and list_posterior[1]>list_posterior[0]):
        if str[1] == kfold_test[i][4]:
            True_count += 1
          
        else:
            False_count += 1
            list_Fp[1]+=1
            if kfold_test[i][4]==str[0]:
                list_Fn[0]+=1
            elif kfold_test[i][4]==str[2]:
                list_Fn[2]+=1

    elif (list_posterior[2]>list_posterior[1] and list_posterior[2]>list_posterior[0]):
        if str[2] == kfold_test[i][4]:
            True_count += 1
          
        else:
            False_count += 1
            list_Fp[2]+=1
            if kfold_test[i][4]==str[0]:
                list_Fn[0]+=1
            elif kfold_test[i][4]==str[1]:
                list_Fn[1]+=1

    else: print("error")

print("True_count:: %d"% True_count)
print("False_count:: %d"% False_count)       

list_precision=[0,0,0]
list_accurancy=[0,0,0]
list_recall=[0,0,0]
for i in range(0,3):
    print(str[i],"case에서")
    list_accurancy[i]=(True_count+list_Tn[i])/(False_count+True_count)   
    print("accurancy::",list_accurancy[i])

    list_precision[i]=True_count/(True_count+list_Fp[i])
    print("precision:",list_precision[i])

    list_recall[i]=True_count/(True_count+list_Fn[i])
    print("recall:",list_recall[i])

    F1_score=2/(1/list_precision[i]+1/list_recall[i])
    print("F1 score:",F1_score)
    print()


#B_4


print("\n\n\n")
print("B-4--------------------------------------------------------------------------------")



list_mean=[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]#첫번째 인덱스 각 feature 두번째 인덱스 각 class
count_c=[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

list_varience=[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

varience=0

for parameter in range(0,4):# 정규분포 mean 구하기
    Matrix = set_Matrix(parameter)
    mean = 0
    for i in range(0, 120):
        for c in range(0, 3):
            if str[c] == kfold_training[i][4]:
                list_mean[parameter][c] += kfold_training[i][parameter]
                count_c[parameter][c]+=1
                break
            else: continue


print("feature 와 class 별 mean::-----------------")

for parameter in range(0,4):# 정규분포 mean 더한거 개수로 나누기
    for c in range(0,3):
        list_mean[parameter][c]/=count_c[parameter][c]
   
    print(list_mean[parameter])


for parameter in range(0,4):# 정규분포 mean 구하기
    Matrix = set_Matrix(parameter)
    for i in range(0, 120):
        for c in range(0, 3):
            if str[c] == kfold_training[i][4]:
                list_varience[parameter][c] += (kfold_training[i][parameter]-list_mean[parameter][c])**2
                break
            else: continue
print()
print("feature 와 class 별 varience::-----------------")
for parameter in range(0, 4):# 정규분포 varience 더한거 개수로 나누기
    for c in range(0, 3):
        list_varience[parameter][c] /= count_c[parameter][c]
        list_varience[parameter][c]=list_varience[parameter][c]**0.5
    print(list_varience[parameter])
print()
def gausian_pdf(mean,deviation,x):
    kf = np.exp(-((x-mean) ** 2) / (2 * deviation ** 2)) / (deviation* math.sqrt(2 *math.pi))
    return kf

def set_likelihood_p4(test,cn):
    return_val = 1
    for parameter in range(0, 4):  # feature 4개돌리기
        sump = 0
        for c in range(0,3):
            sump+=gausian_pdf(list_mean[parameter][c],list_varience[parameter][c],kfold_test[test][parameter])
        return_val*=(gausian_pdf(list_mean[parameter][cn], list_varience[parameter][cn], kfold_test[test][parameter])/sump)
    return return_val

True_count=0; False_count=0

list_prior = [0, 0, 0, 0]

list_prior=set_prior()

list_precision=[0,0,0]
list_accurancy=[0,0,0]
list_recall=[0,0,0]
list_Fp=[0,0,0]
list_Fn=[0,0,0]
list_Tn=[0,0,0]


for test in range(0,30):#test set 돌리기
    list_posterior = [1, 1, 1]
    for cn in range(0, 3):
        list_posterior[cn] = set_likelihood_p4(test, cn)# / list_prior[j]

    if list_posterior[0]>list_posterior[1] and list_posterior[0]>list_posterior[2]:
        if str[0]==kfold_test[test][4]:
            True_count+=1
        else:
            False_count+=1
            list_Fp[0]+=1
            if kfold_test[test][4]==str[1]:
                list_Fn[1]+=1
            elif kfold_test[test][4]==str[2]:
                list_Fn[2]+=1

    elif (list_posterior[1]>list_posterior[2] and list_posterior[1]>list_posterior[0]):
        if str[1] == kfold_test[test][4]:
            True_count += 1
        else:
            False_count += 1
            list_Fp[1]+=1
            if kfold_test[test][4]==str[2]:
                list_Fn[2]+=1
            elif kfold_test[test][4]==str[0]:
                list_Fn[0]+=1
    elif (list_posterior[2]>list_posterior[1] and list_posterior[2]>list_posterior[0]):
        if str[2] == kfold_test[test][4]:
            True_count += 1
        else:
            False_count += 1
            list_Fp[2]+=1
            if kfold_test[test][4]==str[1]:
                list_Fn[1]+=1
            elif kfold_test[test][4]==str[0]:
                list_Fn[0]+=1
    else: print("error")

print("True_count:: %d"% True_count)
print("False_count::%d"% False_count)

for i in range(0,3):
    print(str[i],"case에서")
    list_accurancy[i]=(True_count+list_Tn[i])/(False_count+True_count)   
    print("accurancy::",list_accurancy[i])

    list_precision[i]=True_count/(True_count+list_Fp[i])
    print("precision:",list_precision[i])

    list_recall[i]=True_count/(True_count+list_Fn[i])
    print("recall:",list_recall[i])

    F1_score=2/(1/list_precision[i]+1/list_recall[i])
    print("F1 score:",F1_score)
    print()

"""




"""
#B_5


print("\n\n\n")
print("B-5--------------------------------------------------------------------------------")


list_mean_feature=np.zeros((4,3))#feature , class

count_feature=np.zeros((4,3))


for i in range(0,120):
    for k in range(0,3):#class별
        if(kfold_training[i][4]==str[k]):
            for j in range(0,4):#feature별
                list_mean_feature[j][k]+=kfold_training[i][j]
                count_feature[j][k]+=1

for i in range(0,4):#feature
    for j in range(0,3):#class
        list_mean_feature[i][j]/=count_feature[i][j]

list_sigma_cov=[]
sigma_cov=np.zeros((4,4))

def cov(x,y,EX,EY,cn):# j,i는 parameter 번호
    val=0
    count=0
    for k in range(0,120):
            if  kfold_training[k][4]==str[cn]:
                val+=(kfold_training[k][x]-EX)*(kfold_training[k][y]-EY)
                count+=1
    val/=count
    return val

#class 별로

list_c0=[]
list_c1=[]
list_c2=[]

for t in range(0,120):
    for i in range(0,3):
        if(kfold_training[t][4]==str[0]):
            list_c0.append(kfold_training[k])
        elif kfold_training[t][4]==str[1]:
            list_c1.append(kfold_training[k])
        elif kfold_training[t][4]==str[2]:
            list_c2.append(kfold_training[k])

for c in range(0,3):
    for i in range(0,4):# first feature 
        for j in range(0,4):#second feature
            sigma_cov[j][i]=cov(j,i,list_mean_feature[j][c],list_mean_feature[i][c],c)
    list_sigma_cov.append(sigma_cov)  

def px(c,x):
    p=(1/(2*math.pi)**2*np.linalg.det(list_sigma_cov[c])**2)
    #p*=math.exp(-1/2*np.transpose(subMatrix(x,list_mean_feature[c])* np.inv(list_sigma_cov[c]))*subMatrix(x,list_mean_feature[c])

    M=np.zeros([4,1])
    for i in range(0,4):
        M[i]=list_mean_feature[i][c]
    
    x_M=np.array(x)-np.array(M)
    #print (np.shape(x))
    #print (np.shape(M))
  

    X=np.transpose(x_M)
    #print(np.shape(X))
    #print (np.shape(list_sigma_cov[c]))
    inv_sig=np.linalg.inv(list_sigma_cov[c])
    X=np.dot(X,inv_sig)
    X=np.dot(X,x_M)

    p*=math.exp(-1/2*X)
    return p

True_count=0
False_count=0

list_precision=[0,0,0]
list_accurancy=[0,0,0]
list_recall=[0,0,0]
list_Fp=[0,0,0]
list_Fn=[0,0,0]
list_Tn=[0,0,0]
list_px=[0,0,0]


for test in range(0,30):#test set 돌리기

    x=np.zeros([4,1])
    for i in range(0,4):
        x[i][0]=kfold_test[test][i]
    #print (kfold_test[0][0])
   
    for cn in range(0, 3):
        
        list_px[cn] = px(cn,x)# / list_prior[j]

    if list_px[0]>list_px[1] and list_px[0]>list_px[2]:
        if str[0]==kfold_test[test][4]:
            True_count+=1
        else:
            False_count+=1
            list_Fp[0]+=1
            if kfold_test[test][4]==str[1]:
                list_Fn[1]+=1
            elif kfold_test[test][4]==str[2]:
                list_Fn[2]+=1
    elif list_px[1]>list_px[2] and list_px[1]>list_px[0]:
        if str[1] == kfold_test[test][4]:
            True_count += 1
        else:
            False_count += 1
            list_Fp[1]+=1
            if kfold_test[test][4]==str[0]:
                list_Fn[0]+=1
            elif kfold_test[test][4]==str[2]:
                list_Fn[2]+=1
    elif list_px[2]>list_px[1] and list_px[2]>list_px[0]:
        if str[2] == kfold_test[test][4]:
            True_count += 1
        else:
            False_count += 1
            list_Fp[2]+=1
            if kfold_test[test][4]==str[1]:
                list_Fn[1]+=1
            elif kfold_test[test][4]==str[0]:
                list_Fn[0]+=1
    else: print("error")

print("True_count:: %d"% True_count)
print("False_count::%d"% False_count)


for i in range(0,3):
    print(str[i],"case에서")
    list_accurancy[i]=(True_count+list_Tn[i])/(False_count+True_count)   
    print("accurancy::",list_accurancy[i])

    list_precision[i]=True_count/(True_count+list_Fp[i])
    print("precision:",list_precision[i])

    list_recall[i]=True_count/(True_count+list_Fn[i])
    print("recall:",list_recall[i])

    F1_score=2/(1/list_precision[i]+1/list_recall[i])
    print("F1 score:",F1_score)
    print()