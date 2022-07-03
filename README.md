# Computer-Application-Probability
3-2 학기 이상철교수님 컴퓨터응용확률 iris data 분석 과제 
데이터 분류 


## Data
- csv_all.csv에 150개의 3종류의 꽃이 있다. 
꽃 data 에는 꽃종류를 포함한 5개의 매개 변수가 존재한다.

![6](https://user-images.githubusercontent.com/26202424/177027034-b84aa867-c1aa-4e3f-94db-683a215159de.png)


## k-fold cross-validation method
- data양이 150개로 적기떄문에 데이터를 k fold로 나누고 교차검증해야한다. k의 값에따라 데이터량을 늘릴수 있다.


overfitting될수 있지만 validation dataset과 test dataset의 결과가 비슷하다면 모델을 더 신뢰할수있다.

## data graph


![image](https://user-images.githubusercontent.com/26202424/177029047-65dc212e-3cda-4988-87a8-bf1e566525cc.png)


- 일차원 데이터 그래프를 보면 setosa 꽃이 다른 두종류의 꽃보다 p_length,p_width에서 확연히 다른 길이를 보여준다
- verginica와 versicolor 꽃들도 p_length,p_width에서 비교적 잘 분류할수있다.

## 1차원 그래프에서 오차합을 통한 boundary 찾기

- 각 그래프에서 선 2개를 그어 3종류의 꽃의 종류를 맞춘다고 할때 선 2개의 위치를 계속 움직이면서 error의 총합이 가장 적은선을 구한다.


-  두 선으로 맨왼쪽이 setosa 중간이 versicolor 맨오른쪽에 virginica라고 할떄 
- Error = Count ( class ≠ setosa | predict = setosa ) + 
             Count ( class ≠ Versiclolor | predict = Versicolor) + 
             Count ( class ≠ Virginica | predict = Virginica) 이다. 
             
             
![8](https://user-images.githubusercontent.com/26202424/177028069-0029d388-2c36-431c-b13c-929d103f7a66.png)
            



## 2차원 data graph
![55](https://user-images.githubusercontent.com/26202424/177028422-e361fc05-ad6a-4d86-8d30-d45c8e2d1323.png)

## 2차원 그래프에서 optimal decision boundary 찾기
- logisic regression 사용 분류 
- If h(x)>=0, predict “y=1”
  If h(x)<0.5, predict “y=0”으로 classification한다고 할 때 0<=h(x)<=1을 원합니다. 그래서 우리는 h(x)에 sigmoid 함수를 취해줍니다.
- sigmoid 함수 특성은 아무리 작아도 0이상이고 아무리 커도 1이하이기 떄문입니다. 이 함수의 특성을 활용해서 로지스틱 분류를 한다.
  
![image](https://user-images.githubusercontent.com/26202424/177028775-12c227a9-fe65-4625-9520-194e20d61391.png)
![image](https://user-images.githubusercontent.com/26202424/177028777-97676358-6970-4611-89ec-c8bca04f80f0.png)

- 가설함수를 다음과 같이 설정했습니다.
H(x,y)=Sigmoid(1+w0*x+W1*y)  상수를 1로 설정하는 이유는 w0와 w1에 의해서 실질적으로 y=ax+b라 할떄 a와 b가 정해지기 떄문 입니다.
decision boundary에서 if h(x)>=0일떄 y=1로 예측하고
decision boundary에서 if h(x)<=0일떄 y=0로 예측합니다.

여기서 비용함수는 


![image](https://user-images.githubusercontent.com/26202424/177028798-86d68f62-3829-4e10-b19b-b0a56b1de69b.png)

- cost최소화하는 x,y를 위해서 경사하강법을 사용합니다. log가 없는 cost함수는 h의 도함수가 0인 부분이 많아서 실제로 h(x)가 최소화되는 값이 구해지지 않기떄문이다.
- 이것을 10000번 돌려서 W를 정의 후 h(x)를 구한다.
- multi class선에서 setosa와 다른 2종류, virginica와 다른 2종류, versicolor와 다른 2종류 이렇게 3번 logistic regression을 한다.


![88](https://user-images.githubusercontent.com/26202424/177028454-2588a499-5e41-4530-af22-d29732e15b3d.png)

## 결론
- versicolor와 다른 2종류를 할떄 거의 모든 feature가 setosa , versicolor, virginica 순서대로 있어서 versicolor는 신뢰도가 높지 않다. 
- 그래서 setosa virginica를 분류하는선이 decision boundary이다.


# 3번 부턴 다음에 시간날떄..
