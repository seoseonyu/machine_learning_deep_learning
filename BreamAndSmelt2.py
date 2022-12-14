from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

fish_length = [
    25.4,
    26.3,
    26.5,
    29.0,
    29.0,
    29.7,
    29.7,
    30.0,
    30.0,
    30.7,
    31.0,
    31.0,
    31.5,
    32.0,
    32.0,
    32.0,
    33.0,
    33.0,
    33.5,
    33.5,
    34.0,
    34.0,
    34.5,
    35.0,
    35.0,
    35.0,
    35.0,
    36.0,
    36.0,
    37.0,
    38.5,
    38.5,
    39.5,
    41.0,
    41.0,
    9.8,
    10.5,
    10.6,
    11.0,
    11.2,
    11.3,
    11.8,
    11.8,
    12.0,
    12.2,
    12.4,
    13.0,
    14.3,
    15.0,
]
fish_weight = [
    242.0,
    290.0,
    340.0,
    363.0,
    430.0,
    450.0,
    500.0,
    390.0,
    450.0,
    500.0,
    475.0,
    500.0,
    500.0,
    340.0,
    600.0,
    600.0,
    700.0,
    700.0,
    610.0,
    650.0,
    575.0,
    685.0,
    620.0,
    680.0,
    700.0,
    725.0,
    720.0,
    714.0,
    850.0,
    1000.0,
    920.0,
    955.0,
    925.0,
    975.0,
    950.0,
    6.7,
    7.5,
    7.0,
    9.7,
    9.8,
    8.7,
    10.0,
    9.9,
    9.8,
    12.2,
    13.4,
    12.2,
    19.7,
    19.9,
]

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1] * 35 + [0] * 14

kn = KNeighborsClassifier()

# 훈련 세트로 입력값 중 0부터 34번째 인덱스까지 사용
train_input = fish_data[:35]

# 훈련 세트로 타깃값 중 0부터 34번째 인덱스까지 사용
train_target = fish_target[:35]

# 테스트 세트로 입력값 중 35번째부터 마지막 인덱스까지 사용
test_input = fish_data[35:]

# 테스트 세트로 타깃값 중 35번째부터 마지막 인덱스까지 사용
test_target = fish_target[35:]

kn = kn.fit(train_input, train_target)

score = kn.score(test_input, test_target)
print(score)

# 결과 정확도 0.0 = 샘플링 편향
# 49개의 데이터중 순서대로 35개는 도미 14개는 빙어의 데이터인데 train_input 즉 훈련 데이터에는 도미의 데이터만들어갔으니(편향) 올바르게 도미와 빙어를 분류 할 수 없다.

# 이를 해결하기 위해선 도미와 빙어의 데이터가 골고루 섞여 있어야한다.

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# input_arr와 target_arr 의 데이터는 매칭이되어야한다 하지만 랜덤으로 섞는과정에서 뒤틀릴 수 있으니 random.seed() 를 사용하여 일정한 랜덤 결과를 얻는다.

np.random.seed(42)

index = np.arange(49)
# arrange() 함수에 정수 N을 전달하면 0에서부터 N-1까지 1씩 증가하는 배열을 만든다.

np.random.shuffle(index)
# shuffle() 함수는 주어진 배열을 무작위로 섞는다.

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
# 넘파이 배열 인덱싱을 사용하여 input_arr, target_arr 데이터를 동일한 위치로 섞는다.

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# plt.scatter(train_input[:, 0], train_input[:, 1])
# plt.scatter(test_input[:, 0], test_input[:, 1])
# plt.xlabel("length")
# plt.ylabel("weight")
# plt.show()

# 산점도를 그려 훈련 세트와 테스트 세트에 도미와 빙어가 잘 섞여 있는지 확인한다.
# 파란색이 훈련 세트이고 주황색이 테스트 세트이다.

kn = kn.fit(train_input, train_target)
# 앞서 생성한 훈련 세트로 k-최근접 이웃 모델을 훈련시킨다.

score = kn.score(test_input, test_target)
print(score)
# 테스트 세트로 모델을 테스트한다.

predict_result = kn.predict(test_input)
print(predict_result)
print(test_target)
# predict() 메서드로 테스트 세트의 예측 결과와 실제 타깃을 확인
