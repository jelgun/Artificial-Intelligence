import numpy as np
from collections import defaultdict 
from functools import cmp_to_key

FEATURE_COSTS = [1] * 16 + [22.78, 11.41, 14.51, 11.41, 0]


class DecisionTree:
    def __init__(self, min_impurity_decrease):
        self.min_impurity_decrease = min_impurity_decrease
        self.ruleset = []
        self.extracted_features = []
    
    
    def fit(self, X, y):
        """Creates a decision tree
        Args:
            X: 2d array of feature values
            y: class labels
        Returns:
            None 
        """
        self.ruleset = self.create_decision_tree(X, y, [], len(X))
        
    
    def predict(self, X):
        """Predicts classes of the given samples
        Args:
            X: array of samples
        Returns:
            array: class labels of samples
        """
        self.ruleset.sort(key=len, reverse=True)
        pred = []
        costs = []
        for sample in X:
            for rule, label in self.ruleset:
                holds = True
                distinct_features = set()
                for feature, feature_val, cond in rule:
                    distinct_features.add(feature)
                    if cond == 0:
                        if sample[feature] > feature_val:
                            holds = False
                            break
                    else:
                        if sample[feature] <= feature_val:
                            holds = False
                            break
                if holds:
                    cost = 0
                    for i in distinct_features:
                        if i == 20:
                            if 19 not in distinct_features:
                                cost += FEATURE_COSTS[19]
                            if 18 not in distinct_features:
                                cost += FEATURE_COSTS[18]
                        else:
                            cost += FEATURE_COSTS[i]
                    
                    costs.append(cost)
                    pred.append(label)
                    break
                    
        return pred, costs
    
    
    def entropy_score(self, y):
        """Calculate the entropy of a node
        Args:
            y: array that holds class labels
        Returns:
            float: entropy score 
        """
        freq = defaultdict(lambda:0)
        for i in y:
            freq[i] += 1

        entropy = 0
        total = len(y)
        for key, val in freq.items():
            prob = val / total
            entropy -= prob * np.log(prob)
        
        return entropy


    def split_score(self, feature_index, feature, y, split_val):
        """Calculate the gain of a split
        Args:
            feature: values of the splitted feature
            y: labels of that feature
            split_val: value for split condition
        Returns:
            float: split gain
        """
        
        false_part = []
        true_part = []

        for i in range(len(feature)):
            if feature[i] <= split_val:
                true_part.append(y[i])
            else:
                false_part.append(y[i])

        true_prob = len(true_part) / len(y)
        false_prob = len(false_part) / len(y)
        
        entropy_before = self.entropy_score(y)
        entropy_after = true_prob*self.entropy_score(true_part) + false_prob*self.entropy_score(false_part)
        gain = entropy_before - entropy_after
        
        cost = 1
        if feature_index not in self.extracted_features:
            cost = FEATURE_COSTS[feature_index]
            if feature_index == 20:
                if 18 not in self.extracted_features:
                    cost += FEATURE_COSTS[18]
                if 19 not in self.extracted_features:
                    cost += FEATURE_COSTS[19]
                if cost==0:
                    cost = 1
                
        return gain**2 / cost


    def split_node(self, X, y):
        """Split the node into two
        Args:
            X: feature values
            y: labels
        Returns:
            int: chosen feature
            float: chosen split value
            2d array: true condition feature values
            1d array: true condition labels
            2d array: false condition feature values
            1d array: false condition labels
        """
        max_score = -np.inf
        chosen_feature = 0
        chosen_split_val = 0

        for i in range(X.shape[1]):
            vals = sorted(np.unique(X[:,i]))
            for j in range(len(vals) - 1):
                split_val = (vals[j] + vals[j + 1]) / 2
                score = self.split_score(i, X[:,i], y, split_val)
                if score > max_score:
                    max_score = score
                    chosen_feature = i
                    chosen_split_val = split_val

        true_X = []
        true_y = []
        false_X = []
        false_y = []
        for i in range(len(X)):
            if X[i][chosen_feature] <= chosen_split_val:
                true_X.append(X[i])
                true_y.append(y[i])
            else:
                false_X.append(X[i])
                false_y.append(y[i])

        true_X = np.array(true_X)
        false_X = np.array(false_X)
        
        return chosen_feature, chosen_split_val, true_X, true_y, false_X, false_y

    
    def get_most_frequent(self, y):
        """find the most frequent class label
        Args:
            y: labels
        Returns:
            int: label
        """
        mx = 0
        mx_label = 0
        for i in np.unique(y):
            count = y.count(i)
            if count > mx:
                mx = count
                mx_label = i
                
        return mx_label
    
    
    def create_decision_tree(self, X, y, cur_rule, N):
        """Recursively creates a decision tree
        Args:
            X: feature values
            y: labels
            cur_rule: current rule of a node
            N: total number of samples
        Returns:
            array: the rules of leaf nodes
        """
        if (len(np.unique(y)) == 1):
            # pure node
            return [(cur_rule, y[0])]

        chosen_feature, chosen_split_val, true_X, true_y, false_X, false_y = self.split_node(X, y)

        true_prob = len(true_y) / len(y)
        false_prob = len(false_y) / len(y)
        impurity_before = self.entropy_score(y)
        impurity_after = true_prob*self.entropy_score(true_y) + false_prob*self.entropy_score(false_y)
        
        imp_decrease = (len(X) / N) * (impurity_before - impurity_after)
        if imp_decrease < self.min_impurity_decrease:
            # prune
            return [(cur_rule, self.get_most_frequent(y))]
        
        if chosen_feature not in self.extracted_features:
            self.extracted_features.append(chosen_feature)
            
        if chosen_feature == 20:
            # extract 18th and 19th features
            if 18 not in self.extracted_features:
                self.extracted_features.append(18)
            if 19 not in self.extracted_features:
                self.extracted_features.append(19)
        
        true_rule = cur_rule + [(chosen_feature, chosen_split_val, 0)]
        false_rule = cur_rule + [(chosen_feature, chosen_split_val, 1)]

        # revursevily create tree for left(true) and right(false) nodes
        ruleset = []
        ruleset += self.create_decision_tree(true_X, true_y, true_rule, N)
        ruleset += self.create_decision_tree(false_X, false_y, false_rule, N)

        return ruleset


def accuracy(y_true, y_pred):
    """Calculate accuracy
    Args:
        y_true: true labels
        y_pred: predicted labels
    Returns:
        float: accuracy
    """
    count_true = sum([1 for i in range(len(y_pred)) if y_pred[i] == y_true[i]])
    return count_true / len(y_pred)

def cross_val(min_impurity_dec, k, X, y):
    """Calculate mean cross validation accuracy given model parameter
    Args:
        min_impurity_dec: value of minimum impurity decrease to stop growing the tree
        k: number of cross validations
        X: sample data
        y: labels
    Returns:
        float: mean cross validation accuracy
    """
    size = int(len(X) / k)
    total_acc = 0
    for i in range(k):
        begin = size * i
        end = size * (i + 1)
        if i == k - 1:
            # last set takes all the remaining elements
            end = len(X)
            
        X_val = X[begin:end]
        y_val = y[begin:end]
        X_train = np.concatenate((X[:begin], X[end:]), axis=0) # X is a numpy array
        y_train = y[:begin] + y[end:]
        
        decision_tree = DecisionTree(min_impurity_decrease=min_impurity_dec)
        decision_tree.fit(X_train, y_train)
        y_pred = decision_tree.predict(X_val)[0]
        total_acc += accuracy(y_val, y_pred)
        
    return total_acc / k


def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix and class based accuracies
    Args:
        y_true: true labels
    Returns:
        2d array: confusion matrix
        1d array: precisions
        1d array: recalls
    """
    confusion = np.zeros((3, 3), dtype=np.int32)

    for i in range(len(y_true)):
        true_class = int(y_true[i])
        pred_class = int(y_pred[i])
        confusion[true_class-1][pred_class-1] += 1

    precisions = []
    recalls = []
    for i in range(3):
        precisions.append(confusion[i][i] / sum(confusion[:, i]))
        recalls.append(confusion[i][i] / sum(confusion[i]))
        
    return confusion, precisions, recalls


def report_avg_costs(y_true, costs):
    freq = defaultdict(lambda:0)
    total_cost = defaultdict(lambda:0)
    for i in range(len(y_true)):
        freq[y_true[i]] += 1
        total_cost[y_true[i]] += costs[i]
    
    for i in range(1, 4):
        print("Class", i, "avg. cost:", total_cost[i] / freq[i])
    

def read_data(filepath):
    """Read file the given data file
    Args:
        filepath: path of the file
    Returns:
        2d array: feature values
        1d array: labels
    """
    X = []
    y = []
    data = open(filepath).readlines()
    for i in data:
        row = i.strip().split()
        row = [float(i) for i in row]

        X.append(row[:-1])
        y.append(int(row[-1]))

    X = np.array(X)
    return X, y


X_train, y_train = read_data("ann-train.data")
X_test, y_test = read_data("ann-test.data")


max_acc = 0
for min_impurity_dec in [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0]:
    acc = cross_val(min_impurity_dec, 5, X_train, y_train)
    if acc > max_acc:
        max_acc = acc
        best_param = {
            "min_impurity_dec": min_impurity_dec
        }
        
print("Best parameter:")
print(best_param)


decision_tree = DecisionTree(min_impurity_decrease=0.0005)
decision_tree.fit(X_train, y_train)
train_acc = accuracy(y_train, decision_tree.predict(X_train)[0])
test_acc = accuracy(y_test, decision_tree.predict(X_test)[0])
train_confusion, train_precisions, train_recalls = confusion_matrix(y_train, decision_tree.predict(X_train)[0])
test_confusion, test_precisions, test_recalls = confusion_matrix(y_test, decision_tree.predict(X_test)[0])

print("Train set accuracy:", train_acc)
print("Test set accuracy:", test_acc)
print("Train set confusion matrix:\n", train_confusion)
print("Train set precisions:", train_precisions)
print("Train set recalls:", train_recalls)
print("Test set confusion matrix:\n", test_confusion)
print("Test set precisions:", test_precisions)
print("Test set recalls:", test_recalls)

print("Average cost/class:")
report_avg_costs(y_test, decision_tree.predict(X_test)[1])
