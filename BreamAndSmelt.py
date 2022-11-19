import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

bream_length = [
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
]


bream_weight = [
    242.0,
    290.0,
    340.0,
    363.0,
    430.0,
    350.0,
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
    950,
]

smelt_length = [
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

smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 그래프 출력 테스트
# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel("length")
# plt.ylabel("weight")
# plt.show()

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length, weight)]

# 도미1 빙어0
fish_target = [1] * 35 + [0] * 14

kn = KNeighborsClassifier()

kn.fit(fish_data, fish_target)

# 정확도 확인
print(kn.score(fish_data, fish_target))

# 길이30 무게600의 생선이 무엇인지
# print(kn.predict([[30, 600]]))

kn49 = KNeighborsClassifier(n_neighbors=49) # 참고 데이터를 49개로 한 kn49 모델

kn49.fit(fish_data, fish_target)
print(kn49.score(fish_data, fish_target))

# KNeighborsClassifier 클래스의 기본값은 5이다.
# 사이킷런의 k-최근접 이웃 알고리즘은 주변에서 가장 가까운 5개의 데이터를 보고 다수결의 원칙에 따라 데이터를 예측한다.
# 위의 코드는 해당 값을 49로 즉 가장 가까운 데이터 49개를 사용하는 k-최근접 이웃 모델이 되는것이다.
# 따라서 현재 fish_data의 데이터 49개 중에 도미가 35개로 다수를 차지하므로 어떤 데이터를 넣어도 무조건 도미로 예측할 것 이다.

