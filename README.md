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

![1](https://user-images.githubusercontent.com/26202424/177026961-ea69e116-99bd-4493-a432-74e791d05c1a.png)
![2](https://user-images.githubusercontent.com/26202424/177026963-4290669d-c3fc-4275-b3cf-9582605d00c4.png)
![3](https://user-images.githubusercontent.com/26202424/177026964-71f807ce-cfd1-4da4-b554-07f2c7fdb41e.png)
![4](https://user-images.githubusercontent.com/26202424/177026965-0d44a573-6986-4411-baa5-260f874ebaea.png)

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
![1](https://user-images.githubusercontent.com/26202424/177027637-ddde0085-c126-4f82-9859-53a7c22ca20c.png)
![2](https://user-images.githubusercontent.com/26202424/177027639-169934e4-726d-40ca-a3cb-b8280a3b3d54.png)
![3](https://user-images.githubusercontent.com/26202424/177027641-f6d554fd-d1b1-4f9c-b778-119becbc03f6.png)
![4](https://user-images.githubusercontent.com/26202424/177027642-935c4e75-2a26-46c4-89f1-dd5b4456d3b1.png)
![5](https://user-images.githubusercontent.com/26202424/177027643-768db22d-18ea-4425-9309-d31514c58cd3.png)
![6](https://user-images.githubusercontent.com/26202424/177027644-f438086c-fb6a-4cf5-949c-974e1f758b9b.png)

