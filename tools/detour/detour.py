import random
import argparse
import os
import competition_pb2_grpc
import competition_pb2
import grpc
import concurrent.futures as fut

import numpy as np
import numpy.random as ra
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist

# Curvature


def adjust(thetadiff):
    """This is to restrict the angular difference between -pi and pi"""
    # https://rosettacode.org/wiki/Angle_difference_between_two_bearings
    diff = (thetadiff) % (2 * np.pi)
    if diff >= np.pi:
        diff -= 2 * np.pi
    return diff


vecadjust = np.vectorize(adjust)


def xy2ka(xs, ys):
    """ Convert from (x,y) coordinates to kappa and arclength of pieces
    the number of kappas is 2 short of the number of coordinates
    the number of pieces is 1 short of the number of coordinates"""
    xdiffs = np.diff(xs)
    ydiffs = np.diff(ys)

    thetas = np.atan2(ydiffs, xdiffs)
    thetadiffs = vecadjust(np.diff(thetas))
    arclengths = np.sqrt(xdiffs ** 2 + ydiffs ** 2)
    kappas = thetadiffs / arclengths[:-1]

    return thetas[0], kappas, arclengths


# Approximation

def approximate(kappas, arclengths, N):
    """Reduces given lists of kappas and arclengths to shorter
    versions by approximation."""
    # archlengths' size can be 1 larger than that of kappas
    kappas_list = [float(kappas[i]) for i in range(kappas.shape[0])]
    arclengths_list = [float(arclengths[i]) for i in range(kappas.shape[0])]
    all_kappas_list = [[k] for k in kappas_list]
    all_arclengths_list = [[a] for a in arclengths_list]

    # print(kappas_list)
    # print(arclengths_list)
    while len(kappas_list) > N:
        cur_len = len(kappas_list)
        kappa_avgs = []
        arclength_sums = []
        errors = []
        for j in range(cur_len - 1):
            arclength_sum = arclengths_list[j] + arclengths_list[j + 1]
            kappa_avg = (kappas_list[j] * arclengths_list[j] + kappas_list[j + 1] * arclengths_list[j + 1])/ arclength_sum
            error = sum([np.abs(kappa_avg - k) * a for (k,a) in zip(all_kappas_list[j], all_arclengths_list[j])]) + sum([np.abs(kappa_avg - k) * a for (k,a) in zip(all_kappas_list[j + 1], all_arclengths_list[j + 1])])
            kappa_avgs.append(kappa_avg)
            arclength_sums.append(arclength_sum)
            errors.append(error)

        merge_index = int(np.argmin(errors))
        # print(f"Merge_index: {merge_index}, Merging {kappas_list[merge_index]} and {kappas_list[merge_index+1]}")
        new_kappas_list = []
        new_arclengths_list = []
        new_all_kappas_list = []
        new_all_arclengths_list = []
        for j in range(merge_index):
            new_kappas_list.append(kappas_list[j])
            new_arclengths_list.append(arclengths_list[j])
            new_all_kappas_list.append(all_kappas_list[j])
            new_all_arclengths_list.append(all_arclengths_list[j])

        new_kappas_list.append(kappa_avgs[merge_index])
        new_arclengths_list.append(arclength_sums[merge_index])
        new_all_kappas_list.append(all_kappas_list[merge_index] + all_kappas_list[merge_index + 1])
        new_all_arclengths_list.append(all_arclengths_list[merge_index] + all_arclengths_list[merge_index + 1])

        for j in range(merge_index + 2, len(kappas_list)):
            new_kappas_list.append(kappas_list[j])
            new_arclengths_list.append(arclengths_list[j])
            new_all_kappas_list.append(all_kappas_list[j])
            new_all_arclengths_list.append(all_arclengths_list[j])

        kappas_list = new_kappas_list
        arclengths_list = new_arclengths_list
        all_kappas_list = new_all_kappas_list
        all_arclengths_list = new_all_arclengths_list

        # print(kappas_list)
        # print(arclengths_list)
    return kappas_list, arclengths_list


# Features


def extract_features(xlist, ylist, N):
    """Given a list of coordinates (via xlist and ylist),
    extract 2 * N values representing curvatures and arclengths corresponding to
    those curvatures"""
    t0, k, a = xy2ka(np.array(xlist), np.array(ylist))
    klist, alist = approximate(k, a, N)
    return [t0] + klist + alist


class Road:
    def __init__(self, competition_object, xvalues, yvalues, is_failing, is_selectable, feature_N):
        self.competition_object = competition_object
        self.xvalues = xvalues
        self.yvalues = yvalues
        self.is_failing = is_failing
        self.is_selectable = is_selectable
        self.feature_N = feature_N
        self.features = extract_features(self.xvalues,
                                         self.yvalues,
                                         self.feature_N)


def get_nodes_at_depth(root, depth):
    if depth == 0 or root.count == 1:
        return [root]
    else:
        nodes_from_left = get_nodes_at_depth(root.left, depth - 1)
        nodes_from_right = get_nodes_at_depth(root.right, depth - 1)
        return nodes_from_left + nodes_from_right


def add_info(root, roads, parent=None):
    root.parent = parent
    if root.count == 1: # if node is a leaf
        if roads[root.id].is_failing:
            root.fail_count = 1
        else:
            root.fail_count = 0

        if roads[root.id].is_selectable:
            root.selectable_count = 1
        else:
            root.selectable_count = 0
    else:
        add_info(root.left, roads, root)
        add_info(root.right, roads, root)
        root.fail_count = root.left.fail_count + root.right.fail_count
        root.selectable_count = root.left.selectable_count + root.right.selectable_count


def decrease_selectable_count(node):
    node.selectable_count -= 1
    if node.parent is not None:
        decrease_selectable_count(node.parent)


def get_leafs_of_tree(root):
    if root.count == 1:
        return [root]
    else:
        return get_leafs_of_tree(root.left) + get_leafs_of_tree(root.right)


def get_distance(distance_matrix, n, i, j):
    return distance_matrix[n * i + j - ((i + 2) * (i + 1)) // 2]


def choose_from_selectable_subtree(root, distance_matrix, n):
    leafs = [node for node in get_leafs_of_tree(root)]
    selectables = [node for node in leafs if node.selectable_count == 1]
    failing_oracles = [node for node in leafs if node.fail_count == 1]
    tuples = [(f, s, get_distance(distance_matrix, n, f.id, s.id)) for f in failing_oracles for s in selectables]
    sorted_tuples = sorted(tuples, key=lambda e: e[2])
    return sorted_tuples[0][1]


def select_node(root, distance_matrix, n):
    """Choose a leaf node following the DETOUR TestSelectionPolicy."""
    if root.count == 1:
        return root

    left = root.left
    right = root.right

    # assuming at least left or right is selectable
    ls = left.selectable_count > 0
    rs = right.selectable_count > 0
    lf = left.fail_count > 0
    rf = right.fail_count > 0

    # fail ratios
    lfr = 0
    if lf:
        lfr = left.fail_count / (left.count - left.selectable_count)

    rfr = 0
    if rf:
        rfr = right.fail_count / (right.count - right.selectable_count)

    # if at least on of two subtrees (left or right) is (isFailing, isSelectable)
    # we choose one randomly with probability proportional to (#failOracle/#totalOracle) * isSelectable
    if (lf and ls) or (rf and rs):
        left_value = lfr
        if not ls:
            left_value = 0

        right_value = rfr
        if not rs:
            right_value = 0

        probabilities = [left_value / (left_value + right_value),
                         right_value / (left_value + right_value)]
        new_root = ra.choice([left, right], p = probabilities)
        return select_node(new_root, distance_matrix, n)
    else:
        return choose_from_selectable_subtree(root, distance_matrix, n)


def select_tests(roads, min_ratio):
    features_list =[road.features for road in roads]
    data = np.vstack(features_list)
    dist = pdist(data, 'seuclidean', V=None)
    Z = linkage(dist, method='ward')
    root = to_tree(Z)
    add_info(root, roads)
    selected_nodes = []

    # first min_ratio percentage of roads are those closest to failing oracles
    preselect_count = max([1, int(min_ratio * root.selectable_count)])
    failing_oracle_ids = get_failing_oracle_ids(roads)
    selectable_ids = get_selectable_ids(roads)
    sorted_tuples = sorted([(fid, sid, get_distance(dist, len(roads), fid, sid)) for fid in failing_oracle_ids for sid in selectable_ids], key= lambda e: e[2])
    for i in range(preselect_count):
        selected_id = sorted_tuples[i][1]
        selected_node = find_node_with_id(root, selected_id)
        selected_nodes.append(selected_node)
        decrease_selectable_count(selected_node)

    while True:
        if root.selectable_count == 0:
            break
        selected_node = select_node(root, dist, len(roads))
        selected_nodes.append(selected_node)
        decrease_selectable_count(selected_node)

    return selected_nodes, dist


# breadth first search
def find_node_with_id(root, node_id):
    nodes = [root]
    while len(nodes) > 0:
        node = nodes.pop(0)
        if node.id == node_id:
            return node
        else:
            if node.count > 1:
                nodes.append(node.left)
                nodes.append(node.right)
    return None


def get_failing_oracle_ids(roads):
    return [i for i in range(len(roads)) if roads[i].is_failing]


def get_selectable_ids(roads):
    return [i for i in range(len(roads)) if roads[i].is_selectable]


def get_oracle_ids(roads):
    return [i for i in range(len(roads)) if not roads[i].is_selectable]


def m_closest_oracle_ids(distance_matrix, n, oracle_ids, m, i):
    m = min([len(oracle_ids), m]) # in case m is too large?
    tuples = [(oracle_id, get_distance(distance_matrix, n, i, oracle_id)) for oracle_id in oracle_ids]
    sorted_tuples = sorted(tuples, key=lambda e: e[1])
    return [rtuple[0] for rtuple in sorted_tuples][:m]


def is_m_closest_oracle_all_passing(roads, distance_matrix, n, oracle_ids, m, i):
    ids = m_closest_oracle_ids(distance_matrix, n, oracle_ids, m, i)
    count = 0
    for oid in ids:
        if not roads[oid].is_failing:
            count += 1
    return count >= m


class DETOUR(competition_pb2_grpc.CompetitionToolServicer):
    """
    DETOUR is a test selector.
    """

    def Name(self, request, context):
        return competition_pb2.NameReply(name="DETOUR")

    def Initialize(self, request_iterator, context):
        """Initialize."""
        self.roads = []
        self.N = 6
        self.max_select_ratio = 0.4
        self.min_select_ratio = 0.05
        self.m = 3
        self.w = 4
        for oracle in request_iterator:
            oracle: competition_pb2.Oracle = oracle
            # print("hasFailed={}\ttestId={}".format(oracle.hasFailed, oracle.testCase.testId))
            xvalues = [road_point.x for road_point in oracle.testCase.roadPoints]
            yvalues = [road_point.y for road_point in oracle.testCase.roadPoints]
            self.roads.append(Road(oracle, xvalues, yvalues, oracle.hasFailed, False, self.N))

        return competition_pb2.InitializationReply(ok=True)

    def Select(self, request_iterator, context):
        """bidirectional streaming for high flexibility"""
        for sdc_test_case in request_iterator:
            sdc_test_case: competition_pb2.SDCTestCase = sdc_test_case
            # print("testId={}".format(sdc_test_case.testId))
            xvalues = [road_point.x for road_point in sdc_test_case.roadPoints]
            yvalues = [road_point.y for road_point in sdc_test_case.roadPoints]
            self.roads.append(Road(sdc_test_case, xvalues, yvalues, None, True, self.N))

        selected_nodes, distance_matrix = select_tests(self.roads, self.min_select_ratio)

        min_count = max([1, int(self.min_select_ratio * len(selected_nodes))])
        max_count = max([1, int(self.max_select_ratio * len(selected_nodes))])

        current_count = 0
        questionable_selectable_count = 0
        oracle_ids = get_oracle_ids(self.roads)
        n = len(self.roads)
        for i in range(len(selected_nodes)):
            selected_node = selected_nodes[i]
            current_count += 1
            if is_m_closest_oracle_all_passing(self.roads, distance_matrix, n, oracle_ids, self.m, selected_node.id):
                questionable_selectable_count += 1
            else:
                questionable_selectable_count = 0

            if questionable_selectable_count >= self.w:
                break

        selection_count = min([max_count, max([min_count, current_count])])

        for i in range(selection_count):
            sdc_test_case = self.roads[selected_nodes[i].id].competition_object
            yield competition_pb2.SelectionReply(testId=sdc_test_case.testId)


if __name__ == "__main__":
    print("start test selector")
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port")
    args = parser.parse_args()
    GRPC_PORT = args.port
    GRPC_URL = "[::]:" + GRPC_PORT

    server = grpc.server(fut.ThreadPoolExecutor(max_workers=2))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(DETOUR(), server)

    server.add_insecure_port(GRPC_URL)
    print("start server on port {}".format(GRPC_PORT))
    server.start()
    print("server is running")
    server.wait_for_termination()
    print("server terminated")
