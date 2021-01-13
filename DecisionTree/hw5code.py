import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    index_sort = np.argsort(feature_vector)
    feature_vector = feature_vector[index_sort]
    target_vector = target_vector[index_sort]
    treshold_vector = np.roll(feature_vector, 1)
    good_indicies = feature_vector[1:] != feature_vector[:-1]
    treshold_vector = (treshold_vector + feature_vector) / 2
    treshold_vector = treshold_vector[1:]
    treshold_vector = treshold_vector[good_indicies]
    # print(Counter(feature_vector))
    L_size = np.arange(1, target_vector.size)
    L_1 = np.cumsum(target_vector)
    P1_L = L_1[:-1]/L_size
    P0_L = 1 - P1_L
    H_L = 1 - P0_L*P0_L - P1_L*P1_L
    R_size = target_vector.size - L_size
    R_1 = L_1[-1] - L_1[:-1]
    P1_R = R_1/R_size
    P0_R = 1 - P1_R
    H_R = 1 - P0_R * P0_R - P1_R * P1_R

    ginis_vector = -L_size/target_vector.size
    ginis_vector *= H_L
    temp = R_size/target_vector.size
    temp *= H_R
    ginis_vector -= temp
    ginis_vector = ginis_vector[good_indicies]

    best_index = np.argmax(ginis_vector)
    best_ginis = ginis_vector[best_index]
    best_treshold = treshold_vector[best_index]
    return treshold_vector, ginis_vector, best_treshold, best_ginis

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}
            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError
            if len(np.unique(feature_vector)) == 1:
                continue
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common()[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']
        else:
             current_feature = node['feature_split']
             if self._feature_types[current_feature] == 'real':
                 if x[current_feature] < node['threshold']:
                     return self._predict_node(x, node['left_child'])
                 else:
                     return self._predict_node(x, node['right_child'])
             else:
                 if x[current_feature] in node['categories_split']:
                    return self._predict_node(x, node['left_child'])
                 else:
                    return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def get_params(self, deep=True):
        return {
        'feature_types': self._feature_types,
        'max_depth': self._max_depth,
        'min_samples_split': self._min_samples_split,
        'min_samples_leaf': self._min_samples_leaf
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
