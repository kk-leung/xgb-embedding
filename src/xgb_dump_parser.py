import math
import numpy as np


class DecisionTree:
    def __init__(self, model_dump, feature_count, feature_names=None):
        self.feature_count = feature_count
        self.decision_nodes = {}
        self.leaf_nodes = {}
        self.parent_nodes = {}
        self.feature_names = feature_names
        self.feature_scores = [0] * feature_count

        self.parse(model_dump)

        self.leaf_to_index = {x: i for i, x in enumerate(list(self.leaf_nodes.keys()))}
        self.self_cover = np.zeros((len(self.leaf_to_index), len(self.leaf_to_index)))
        self.self_value = np.zeros((len(self.leaf_to_index), len(self.leaf_to_index)))

        self.get_cover_intersect()
        self.get_pred_diff()

    def parse(self, model_dump):
        lines = model_dump.split("\n")
        for line in lines[:-1]:
            line_split = line.split(":")
            node_id = int(line_split[0].replace("\t", ""))

            if "leaf" in line:
                self.leaf_nodes[node_id] = LeafNode(node_id, line)
            else:
                node = DecisionNode(node_id, line, self.feature_count, self.feature_names)
                self.decision_nodes[node_id] = node
                self.parent_nodes[node.no_node_id] = node_id
                self.parent_nodes[node.yes_node_id] = node_id
                self.feature_scores[node.feature_id] += 1

    def get_parent_id(self, node_id):
        return self.get_or_null(self.parent_nodes, node_id)

    def get_decision_node(self, node_id):
        return self.get_or_null(self.decision_nodes, node_id)

    def get_leaf_node(self, node_id):
        return self.get_or_null(self.leaf_nodes, node_id)

    def get_depth(self, node_id=None, current_depth=None):
        if node_id is None or current_depth is None:
            return self.get_depth(0, 0)
        if node_id in self.decision_nodes:
            node = self.decision_nodes[node_id]
            left_depth = self.get_depth(node.yes_node_id, current_depth + 1)
            right_depth = self.get_depth(node.no_node_id, current_depth + 1)
            return max(left_depth, right_depth)
        elif node_id in self.leaf_nodes:
            return current_depth
        else:
            raise RuntimeError("node {} not found".format(node_id))

    def get_cover(self, node_id):
        if node_id in self.leaf_nodes.keys():
            return self.leaf_nodes[node_id].cover
        elif node_id in self.decision_nodes.keys():
            return self.decision_nodes[node_id].cover
        else:
            raise RuntimeError("node {} not found".format(node_id))

    @staticmethod
    def get_or_null(dic: dict, key):
        return dic[key] if key in dic.keys() else None

    def get_cover_intersect(self):
        self._recursive_cover(0)
        total_cover = self.get_cover(0)
        self.self_cover = (total_cover - self.self_cover) / total_cover

    def _recursive_cover(self, node_id: int):
        if node_id in self.decision_nodes.keys():
            node = self.decision_nodes[node_id]
            left_leaves = self._recursive_cover(node.yes_node_id)
            right_leaves = self._recursive_cover(node.no_node_id)
            for left_leaf in left_leaves:
                for right_leaf in right_leaves:
                    left = self.leaf_to_index[left_leaf]
                    right = self.leaf_to_index[right_leaf]
                    self.self_cover[left, right] = node.cover
                    self.self_cover[right, left] = node.cover
            return left_leaves + right_leaves
        elif node_id in self.leaf_nodes.keys():
            return [node_id]
        else:
            raise RuntimeError("node {} not found".format(node_id))

    def get_pred_diff(self):
        for node_id1 in self.leaf_nodes.keys():
            for node_id2 in self.leaf_nodes.keys():
                left_node = self.leaf_nodes[node_id1]
                right_node = self.leaf_nodes[node_id2]
                left = self.leaf_to_index[node_id1]
                right = self.leaf_to_index[node_id2]
                self.self_value[left, right] = abs(left_node.leaf_value - right_node.leaf_value)


class DecisionNode:
    def __init__(self, node_id, line, feature_count, feature_names):
        open_bracket = line.index("[")
        close_bracket = line.index("]")

        decision_string = line[open_bracket + 1: close_bracket]
        decision_parts = decision_string.split("<")
        feature_name = decision_parts[0]
        feature_id = int(feature_name[1:])

        if feature_id >= feature_count:
            raise RuntimeError("Feature count mismatch.Model contains f{} when it was expecting {} features"
                               .format(feature_id, feature_count))

        if feature_names is not None:
            feature_name = feature_names.get(feature_id)

        split_value = decision_parts[1]
        decision_value = math.inf if split_value is "inf" else float(split_value)

        attributes = line[close_bracket + 2:].split(",")
        self.yes_node_id = int(attributes[0].split("=")[1])
        self.no_node_id = int(attributes[1].split("=")[1])
        self.missing_node_id = int(attributes[2].split("=")[1])
        self.gain = float(attributes[3].split("=")[1])
        self.cover = float(attributes[4].split("=")[1])
        self.node_id = node_id
        self.feature_id = feature_id
        self.decision_value = decision_value
        self.feature_name = feature_name


class LeafNode:
    def __init__(self, node_id, line):
        attributes = line.split(":")[1].split(",")
        self.leaf_value = float(attributes[0].split("=")[1])
        self.cover = float(attributes[1].split("=")[1])
        self.node_id = node_id
