import numpy as np
import math
# Метрики

# NDCG - нормализиированная версия Discounted Cumulative Gain
# DCG - сумма rel / log2(i + 1), где i от 1 до N, N - количество предсказанных объектов; для каждого предсказанного объекта наша релевантность - единица
# Идеальный вариант (IDCG) - где все релевантные объекты идут по порядку в начале списка
# NDCG = DCG / IDCG
# Сама метрика учитывает положение предсказанных объектов и выдаёт значение в пределах [0; 1]
def ndcg(test_items, predicted_items, predicted_scores):
    items = sorted(zip(predicted_items, predicted_scores), key=lambda x: x[1], reverse=True)
    dcg = 0.0
    test_set = set(test_items)
    count_of_hits = 0
    for i, data in enumerate(items, 0):
        prediction, score = data
        if prediction in test_set:
            dcg += 1.0 / math.log(i + 2, 2)
    idcg = 0.0
    ideal_count = len(test_items) if len(test_items) < len(predicted_items) else len(predicted_items)
    for i in range(ideal_count):
        idcg += 1.0 / math.log(i + 2, 2)
    return (dcg / idcg)

# Hit Rate - довольно простая метрика оценки, менее информативная чем NDCG
# Каждый раз когда, объект из предсказанной выборки "попадает" (те реально имеется) в тестовое - засчитываем hit (те +1)
# Ну и нормализуется делением на мощность множества предсказанных объектов
def hr(test_items, predicted_items):
    total = len(test_items) if len(test_items) < len(predicted_items) else len(predicted_items)
    test_items = set(test_items)
    hits = 0
    for item in predicted_items:
        if item in test_items:
            hits += 1
    return float(hits) / total


def flood_negative_samples(train_set, test_set, samples_per_user=5):
    data = train_set.union(test_set)
    user_set = set()
    item_set = set()
    for u, i in data:
        user_set.add(u)
        item_set.add(i)
    items = list(item_set)
    negative_samples = []
    for user in user_set:
        sample = set()
        while len(sample) != samples_per_user:
            item_index = np.random.randint(0, len(item_set))
            item_index = items[item_index]
            pair = (user, item_index)
            if pair not in data and pair not in sample:
                sample.add(pair)
        negative_samples.extend(list(sample))
    return negative_samples