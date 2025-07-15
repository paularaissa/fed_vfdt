import base64
import warnings
import numpy as np
import flwr as fl
import csv
import sys
import grpc
from river import metrics, preprocessing, ensemble, utils, stats
from river.tree.nodes.leaf import HTLeaf
from river.tree import nodes as tree_nodes
from river.tree.nodes.branch import Branch, NominalBinaryBranch, NumericBinaryBranch
from pathlib import Path
from fht import FederatedHoeffdingTree

from flwr.common import NDArrays
import pickle
import os
import json
import uuid
import time

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*DataFrame concatenation with empty.*")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

class FedVFDTClient(fl.client.NumPyClient):
    """Federated Very Fast Decision Tree (Fed-VFDT) client for Flower.

    This client reads data from a CSV file and trains a federated Hoeffding Tree model
    (VFDT) using the River framework. It detects local split conditions, sends
    statistics to the server, and applies global split decisions received from the server
    to maintain consistency across all clients.

    Attributes:
        file_path (str or Path): Path to the client's local CSV dataset.
        logs_path (Path): Directory where client logs are saved.
        client_id (int): Unique identifier for the client.
        model (FederatedHoeffdingTree): The local VFDT model adapted for federated training.
        accuracy (metrics.Accuracy): Metric to track accuracy during training.
        f1 (metrics.MacroF1): Metric to track macro-averaged F1-score.
        kappa (metrics.CohenKappa): Metric to track the Cohen's Kappa score.
        grace_period (int): Minimum number of instances observed between split attempts.
        round (int): Current training round.
        prev_n_nodes (int): Previous number of nodes in the tree, used to detect growth.
        split_info (bool): Indicates whether a new local split was detected.
        global_split_info (dict): Global split information received from the server.
        reader (csv.DictReader): Iterator to read the dataset line-by-line.
        log_writer (csv.DictWriter): Writer to log training metrics per instance.
        start_time (float): Time when training started, used to measure runtime.
        count_instances (int): Number of instances processed so far.
        frozen_since_round (int): Round number since the client was frozen due to pending split.
        last_split_id (str): UUID of the most recent local split node.
        prev_branch_uuids (set): Set of all branch UUIDs before the current round.
        leaf_ids_before (set): Set of leaf node IDs before split detection.
        tree_snapshot (Any): Snapshot of the tree (if used for debugging or comparison).
        n_seen_by_leaf (dict): Map of leaf UUIDs to number of instances seen.
        replay_buffer (list): Buffer of instances for potential redistribution after split.
        logs_all (list): Aggregated logs (optional use).

    Methods:
        set_parameters(parameters): Replaces the local model with the aggregated one, if valid.
        _collect_branch_uuids(node): Recursively collects UUIDs from all branch nodes.
        _detect_new_split(): Detects whether a new split occurred by comparing branch UUIDs.
        read_next_value(): Reads the next instance (x, y) from the CSV dataset.
        close(): Closes the dataset file handle.
        safe_metric_get(metric): Safely retrieves the value of a metric, avoiding NaN/infinity.
        extract_tree_info(): Extracts detailed statistics about the most recent split node.
        debug_print_leaf_ids(): Prints IDs and stats of all leaves for debugging purposes.
        _find_node_by_uuid(node, uuid_target): Finds a node in the tree by its UUID.
        _find_node_by_path(node, path): Finds a node by its condition path.
        send_tree_stats(): Serializes and returns the local split candidate to the server.
        debug_print_leaves(): Prints leaf nodes and internal node stats for inspection.
        _get_leaf_for_instance(node, x): Finds the leaf where an instance would be routed.
        ensure_leaf_has_uuid(leaf): Ensures the leaf node has a UUID.
        build_paths(node, current_path, paths): Builds a path map to each leaf by UUID.
        fit(parameters, config): Main training loop called by Flower in each round.

    Usage:
        client = FedVFDTClient(file_path, logs_path, client_id)
        fl.client.start_client(server_address="127.0.0.1:8083", client=client.to_client())
"""

    def __init__(self, file_path, logs_path, client_id):
        self.file_path = file_path
        self.index = 0  # Row index tracking
        self.server_active = True  # Flag to check if the server is active
        self.client_id = client_id

        # Initializes the Hoeffding Tree Classifier model from River
        self.model = FederatedHoeffdingTree(#grace_period=67,
                                            #grace_period=100,
                                            # grace_period=40,
                                            grace_period=20,
                                            #grace_period=10,
                                            #grace_period=200,
                                            delta=1e-5,
                                            split_criterion="gini",
                                            #nominal_attributes=['protocol_type', 'service', 'flag']
                                            )
        self.accuracy = metrics.Accuracy()
        self.f1 = metrics.MacroF1()
        self.kappa = metrics.CohenKappa()
        self.grace_period = self.model.grace_period
        self.values_plt_all = []
        self.round = 1  # Control the number of rounds
        self.prev_n_nodes = 1
        self.total_clients = 0
        self.split_info = True
        self.global_split_info = {}  # Stores the globally aggregated feature from the server

        self.file = open(self.file_path, "r")
        self.reader = csv.DictReader(self.file)
        self.logs_path = logs_path
        os.makedirs(self.logs_path, exist_ok=True)
        self.log_file_name = logs_path / f"client_{client_id}.csv"
        self.log_file = open(self.log_file_name, mode="a", newline="")
        fieldnames = ["instances", "round", "splits", "depth", "nodes", "leaves", "accuracy", "f1"]
        self.log_writer = csv.DictWriter(self.log_file, fieldnames=fieldnames)
        # Writes header if the file is empty
        if self.log_file.tell() == 0:
            self.log_writer.writeheader()
        self.start_time = time.time()
        self.count_instances = 0
        self.frozen_since_round = None
        self.last_split_id = None
        self.prev_branch_uuids = set()
        self.leaf_ids_before = set()
        self.tree_snapshot = None
        self.n_seen_by_leaf = {}
        self.replay_buffer = []
        self.logs_all = []

    def set_parameters(self, parameters):
        """Decides whether the client uses the aggregated model or keeps its own model."""
        if parameters and len(parameters) > 0:
            try:
                aggregated_model = pickle.loads(parameters[0])
                print("📥 Client received aggregated model from the server.")
                # Check the structure of the received model
                print(f"🔍 Received model: {type(aggregated_model)}")
                if isinstance(aggregated_model, FederatedHoeffdingTree) or isinstance(aggregated_model, ensemble.VotingClassifier):
                    self.model = aggregated_model
                    print("✅ Client adopted the aggregated model.")
                else:
                    print("⚠️ Received model is not valid. Keeping the previous model.")
            except Exception as e:
                print(f"❌ Error deserializing the model received from the server: {e}")
                print("⚠️ Keeping the previous model.")
        else:
            print("⚠️ No model received from the server. Keeping the current model.")

    def _collect_branch_uuids(self, node):
        """Traverses the entire tree, assigns a UUID to each new Branch, and returns the set of UUIDs."""
        uuids = set()
        if isinstance(node, tree_nodes.branch.Branch):
            # Ensures that every Branch has a UUID before adding
            if not hasattr(node, "uuid") or node.uuid is None:
                node.uuid = str(uuid.uuid4())
            uuids.add(node.uuid)
            children = (node.children
                        if isinstance(node.children, list)
                        else list(node.children.values()))
            for child in children:
                uuids |= self._collect_branch_uuids(child)
        return uuids

    def _detect_new_split(self):
        """Compares prev_branch_uuids with the current ones and returns the new UUID (or None)."""
        current = self._collect_branch_uuids(self.model._root)
        added = current - self.prev_branch_uuids
        # Updates to the next round
        self.prev_branch_uuids = current
        if added:
            # If there is more than one, pick one arbitrarily (or choose based on depth)
            return added.pop()
        return None

    def read_next_value(self):
        """Reads the next row from the CSV and returns a dictionary of features (x) and the label (y)."""
        try:
            row = next(self.reader)
            # x = {
            #    "day": float(row["day"]),
            #    "period": float(row["period"]),
            #    "nswprice": float(row["nswprice"]),
            #    "nswdemand": float(row["nswdemand"]),
            #    "vicprice": float(row["vicprice"]),
            #    "vicdemand": float(row["vicdemand"]),
            #    "transfer": float(row["transfer"]),
            # }
            # y = row['class']

            x = {
                #"src_port": float(row["src_port"]),
                #"dst_port": float(row["dst_port"]),
                "ttl": float(row["ttl"]),
                "tcp_window_size": float(row["tcp_window_size"]),
                "eth_size": float(row["eth_size"]),
                "payload_length": float(row["payload_length"]),
                "l4_tcp": float(row["l4_tcp"]),
                "l4_udp": float(row["l4_udp"]),
                "inter_arrival_time": float(row["inter_arrival_time"]),
                "jitter": float(row["jitter"]) if row["jitter"].strip() != "" else 0.0,
                "stream_5_mean": float(row["stream_5_mean"]) if row["stream_5_mean"].strip() != "" else 0.0,
                #"stream_5_var": float(row["stream_5_var"]),
                #"stream_jitter_5_mean": float(row["stream_jitter_5_mean"]),
                #"stream_jitter_5_var": float(row["stream_jitter_5_var"]),
                #"src_ip_5_var": float(row["src_ip_5_var"]),
                "payload_entropy": float(row["payload_entropy"]),
                "dns_interval": float(row["dns_interval"]),
                "dns_len_qry": float(row["dns_len_qry"]),
                "handshake_cipher_suites_length": float(row["handshake_cipher_suites_length"]),
                "handshake_extensions_length": float(row["handshake_extensions_length"]),
            }
            y = row['Label']

            return x, y
        except StopIteration:
            return None, None

    def close(self):
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()

    @staticmethod
    def safe_metric_get(metric):
        try:
            value = metric.get()
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                return float("nan")
            return value
        except Exception as e:
            print(f"⚠️ Error computing metric {metric}: {e}")
            return float("nan")

    def extract_tree_info(self):
        """Retrieves information from the node that was recently split in the decision tree."""

        # If no local split is detected, there is no information to send.
        if self.last_split_id is None:
            self.split_info = False
            return None

        # Finds the branch that just emerged using the local UUID
        node = self._find_node_by_uuid(self.model._root, self.last_split_id)
        if node is None:
            print(f"⚠️ Newly created local branch (uuid={self.last_split_id}) not found in the tree.")
            return None

        # UUID of the node of interest
        leaf_uuid = self.last_split_id

        # Retrieves the information of the node that split
        feature     = getattr(node, "feature", "Unknown")
        threshold   = getattr(node, "threshold", "Unknown")
        repr_split  = getattr(node, "repr_split", "No information")
        stats       = getattr(node, "stats", "No statistics available")
        depth       = getattr(node, "depth", 0)
        split_test  = type(node).__name__

        # Computes the range for the Hoeffding bound
        split_criterion = self.model._new_split_criterion()
        if hasattr(node, "stats"):
            range_val = split_criterion.range_of_merit(node.stats)
        else:
            range_val = 1

        # Total number of examples seen so far
        total_examples_seen = self.model.summary.get("total_observed_weight", 1)
        epsilon = self.model._hoeffding_bound(range_val, self.model.delta, total_examples_seen)

        # Compute Gini Index
        gini_value = None
        if isinstance(stats, dict) and stats:
            try:
                total_samples = sum(stats.values())
                if total_samples > 0:
                    probabilities = [count / total_samples for count in stats.values()]
                    gini_value = 1 - sum(p ** 2 for p in probabilities)
                else:
                    print("⚠️ Total samples in the node is zero. Cannot compute Gini Index.")
            except Exception as e:
                print(f"⚠️ Error computing Gini Index: {e}")

        # For logs
        classes = list(stats.keys()) if isinstance(stats, dict) else []
        print(f"📚 Classes present in the split node: {classes}")
        print(f"📌 Last split detected in the round {self.round}!")
        print(f"🔹 Selected feature: {feature}")
        print(f"🔹 Split threshold: {threshold}")
        print(f"📉 Hoeffding Bound: {epsilon}")
        print(f"📊 Gini Index: {gini_value}")
        print(f"📌 Leaf UUID: {leaf_uuid}")

        # Returns exactly what send_tree_stats and the server expect
        return feature, threshold, repr_split, stats, epsilon, gini_value, leaf_uuid, split_test, depth

    def debug_print_leaf_ids(self):
        def walk(node):
            if getattr(node, "_is_leaf", True):
                print(f"🌿 Leaf: {getattr(node, 'id', 'no-id')} | n={getattr(node, 'n', '?')}")
            elif hasattr(node, "children"):
                for child in node.children:
                    walk(child)
        print("🌳 Current state of the tree:")
        walk(self.model._root)

    def _find_node_by_uuid(self, node, uuid_target):
        """Recursive search for the 'uuid' attribute in all nodes."""
        if getattr(node, "uuid", None) == uuid_target:
            return node
        if hasattr(node, "children"):
            children = (
                node.children.values()
                if isinstance(node.children, dict)
                else node.children
            )
            for child in children:
                found = self._find_node_by_uuid(child, uuid_target)
                if found:
                    return found
        return None

    def _find_node_by_path(self, node, path):
        """Traverses the tree following a path of conditions and returns the final node."""
        if not path:
            return node
        if not hasattr(node, "children"):
            return None

        next_cond = path[0]
        children = node.children if isinstance(node.children, list) else list(node.children.values())

        for i, child in enumerate(children):
            feature = getattr(node, "feature", None)
            threshold = getattr(node, "threshold", None)

            if isinstance(node, tree_nodes.branch.NumericBinaryBranch):
                cond_expected = f"{feature} <= {threshold}" if i == 0 else f"{feature} > {threshold}"
            elif isinstance(node, tree_nodes.branch.NominalBinaryBranch):
                cond_expected = f"{feature} == x" if i == 0 else f"{feature} != x"
            else:
                cond_expected = f"{feature} ?"

            if cond_expected == next_cond:
                return self._find_node_by_path(child, path[1:])

        return None

    def send_tree_stats(self):
        """Serializes and returns the pending split suggestion to the server."""
        pending = self.model._pending_split
        if not pending:
            print("⚠️ No pending split. Nothing to send.")
            return []

        # Extracts everything directly from pending
        leaf_id        = pending["leaf_uuid"]
        feature        = pending["feature"]
        threshold      = pending["threshold"]
        stats          = pending["stats"]
        gini           = pending["gini"]
        epsilon        = pending["epsilon"]
        depth          = pending.get("depth", None)
        split_test_obj = pending["split_test"]

        if hasattr(split_test_obj, "children_stats"):
            print(f"📊 children_stats found: {split_test_obj.children_stats}")
            # Serializes only the split_test
            try:
                split_test_ser = base64.b64encode(pickle.dumps(split_test_obj)).decode("utf-8")
            except Exception as e:
                print(f"❌ Error while serializing split_test: {e}")
                split_test_ser = None

        # Correctly serializes the children_stats
        try:
            if hasattr(split_test_obj, "children_stats"):
                children_stats_serialized = base64.b64encode(pickle.dumps(
                    split_test_obj.children_stats
                )).decode("utf-8")
            else:
                print("⚠️ split_test_obj has no children_stats.")
                children_stats_serialized = None
        except Exception as e:
            print(f"❌ Error serializing children_stats: {e}")
            children_stats_serialized = None

        msg = {
            "client_id":       self.client_id,
            "leaf_id":         leaf_id,
            "feature":         feature,
            "threshold":       threshold,
            "gini_index":      gini,
            "gain_local":      gini,
            "hoeffding_bound": epsilon,
            "node_stats":      stats,
            "split_test":      split_test_ser,
            "depth":           depth,
            "path":            pending["leaf_path"],
            "children_stats":  children_stats_serialized,
        }

        print(f"📤 Sending stats to the server: {msg}")
        return msg


    def debug_print_leaves(self):
        """Prints only the tree leaves, showing id/uuid and stats."""
        def recurse(node, depth=0):
            indent = "    " * depth
            # Identifier: tries uuid, falls back to id() if it doesn't exist
            #node_id = getattr(node, "uuid",
            #                  getattr(node, "id", "<no-id>"))
            # Detects leaf by type
            if isinstance(node, tree_nodes.leaf.HTLeaf):
                if not hasattr(node, "uuid") or node.uuid is None:
                    node.uuid = str(uuid.uuid4())
                stats = getattr(node, "stats", {})
                depth = getattr(node, "depth", 0)
                #splitter = getattr(node, "splitter", None)
                #splitters = getattr(node, "splitters", {})
                total_weight = getattr(node, "total_weight", 0)
                last_split_attempt_at = getattr(node, "last_split_attempt_at", 0)
                print(f"{indent}🌿 Leaf(id={node.uuid}) stats={stats} depth={depth} total_weight={total_weight} last_split_attempt_at={last_split_attempt_at}")
            # Detects internal node
            elif isinstance(node, tree_nodes.branch.Branch):
                feature   = getattr(node, "feature", "<none>")
                threshold = getattr(node, "threshold", "<none>")
                stats     = getattr(node, "stats", {})
                total_weight = getattr(node, "total_weight", 0)
                print(f"{indent}🔹 Node(id={node.uuid}) split on {feature} ≤ {threshold} stats={stats}")
                print(f"{indent} - total_weight: {total_weight}")
                # Children can be a list (binary) or a dict (multinomial)
                children = (node.children if isinstance(node.children, list)
                            else list(node.children.values()))
                for child in children:
                    recurse(child, depth + 1)
            #else:
            # Rare case: neither HTLeaf nor Branch
            #    print(f"{indent}? Node(id={node_id}) unknown type: {type(node)}")
        print("=== Decision tree ===")
        recurse(self.model._root)
        print("=========================")

    def _get_leaf_for_instance(self, node, x):
        """Traverses the tree until it finds the leaf where x falls."""
        if node is None:
            return None

        # If it's a leaf, return
        if isinstance(node, HTLeaf) or getattr(node, "_is_leaf", False):
            return node

        # Numeric branch
        if isinstance(node, tree_nodes.branch.NumericBinaryBranch):
            feature = node.feature
            threshold = node.threshold
            if x.get(feature, 0) <= threshold:
                return self._get_leaf_for_instance(node.children[0], x)
            else:
                return self._get_leaf_for_instance(node.children[1], x)

        # Nominal branch
        if isinstance(node, tree_nodes.branch.NominalBinaryBranch):
            feature = node.feature
            value = node.split_test.value if hasattr(node.split_test, "value") else None
            if x.get(feature) == value:
                return self._get_leaf_for_instance(node.children[0], x)
            else:
                return self._get_leaf_for_instance(node.children[1], x)

        # Safe fallback
        print(f"⚠️ _get_leaf_for_instance returned a non-leaf node: {type(node)}")
        return None


    def ensure_leaf_has_uuid(self,leaf):
        if not hasattr(leaf, 'uuid') or leaf.uuid is None:
            leaf.uuid = str(uuid.uuid4())

    def build_paths(self, node, current_path=None, paths=None):
        if paths is None:
            paths = {}
        if current_path is None:
            current_path = []

        # If it's a leaf, store the path
        if getattr(node, "_is_leaf", True):
            # ensures UUID
            if not hasattr(node, "uuid") or node.uuid is None:
                node.uuid = str(uuid.uuid4())
            paths[node.uuid] = list(current_path)
        else:
            # it's a branch, explore children
            children = node.children if isinstance(node.children, list) \
                else list(node.children.values())
            for i, child in enumerate(children):
                feature   = getattr(node, "feature", None)
                threshold = getattr(node, "threshold", None)
                if isinstance(node, NumericBinaryBranch):
                    cond = f"{feature} <= {threshold}" if i == 0 else f"{feature} > {threshold}"
                else:
                    cond = f"{feature} == x" if i == 0 else f"{feature} != x"
                self.build_paths(child, current_path + [cond], paths)

        return paths


    # CSVClient.fit
    def fit(self, parameters: NDArrays, config):
        # 0) Always print the tree before anything else
        #self.model.debug_print_tree()

        # 1) If an aggregated model has arrived, apply it and print again
        if parameters and len(parameters) > 0:
            try:
                agg = pickle.loads(parameters[0])
                if isinstance(agg, FederatedHoeffdingTree):
                    self.model = agg
                    self.model._frozen = False
                    print("📥 Aggregated model received — learning resumed.")
                    #self.model.debug_print_tree()
            except Exception as e:
                print(f"❌ Error deserializing aggregated model: {e}")

        # 2) If a global split has arrived, apply it and print
        if "global_split_info" in config:
            info = json.loads(config["global_split_info"])
            # print(f"******** info: {info} ***********")
            required_keys = ["feature", "threshold", "path", "split_test"]
            if all(k in info for k in required_keys):
                print("📦 Applying global split from the server.")
                self.model.apply_aggregated_split(info)
                success = self.model.apply_aggregated_split(info)
                if success:
                    print(f"✅ [Client {self.client_id}] Split applied successfully.")
                else:
                    print(f"❌ [Client {self.client_id}] Failed to apply split — path not found.")

    # self.redistribute_buffer_after_split(
                #     split_path=info.get("path"),
                #     feature=info.get("feature"),
                #     threshold=info.get("threshold")
                # )
                #print("***** AFTER SPLIT *****")
                # self.model.debug_print_tree()
                # Reapplies the last real instance, which is not in the buffer
                if success and self.model.last_instance:
                    self.model.learn_one(*self.model.last_instance)

                # Updates last_split_attempt_at again to reflect this increment
                if success:
                    for leaf in self.model._get_all_active_leaves():
                        leaf.last_split_attempt_at = getattr(leaf, "total_weight", 0.0)

                # Clears the buffer after redistribution
                #self.replay_buffer = []

                # print("🔁 Reapplying buffer after federated split...")
                # for x_buf, y_buf in self.replay_buffer:
                #     self.model.learn_one(x_buf, y_buf)
                # Clears the buffer for the next round
                # self.replay_buffer.clear()
                # print("++++ AFTER REAPPLYING THE BUFFER ++++")
                # self.model.debug_print_tree()
                # print("++++++++++++++++++++++++++++++++++++++")
            else:
                print(f"⚠️ Invalid or incomplete global split received. Ignoring. Missing fields: {[k for k in required_keys if k not in info]}")
                self.model._frozen = False
                # Updates last_split_attempt_at of all active leaves
                for leaf in self.model._get_all_active_leaves():
                    leaf.last_split_attempt_at = getattr(leaf, "total_weight", 0.0)
                self.model._pending_split = None
        else:
            print("ℹ️ No global split received this round.")
            self.model._frozen = False
            self.model._pending_split = None
            for leaf in self.model._get_all_active_leaves():
                leaf.last_split_attempt_at = getattr(leaf, "total_weight", 0.0)

        # 3) If frozen awaiting voting, pause
        if getattr(self.model, "_frozen", False):
            print("⏸ Learning paused — waiting for global split.")

            # If there is a pending split, resend it to ensure delivery.
            if self.model._pending_split is not None:
                stats_msg = json.dumps(self.send_tree_stats())
                #self.model._pending_split = None
                #print("📤 Resending pending stats (still waiting for server).")
                return parameters, 1, {
                    "tree_stats": stats_msg,
                    "accuracy": self.safe_metric_get(self.accuracy),
                    "f1": self.safe_metric_get(self.f1),
                    "continue_training": True,
                }
            return parameters, 0, {"continue_training": True}

        # 4) Read and learning loop
        while True:
            x, y = self.read_next_value()
            if x is None:
                end_time = time.time()
                total_time = round(end_time - self.start_time, 4)

                # Saves total execution time in a separate file
                with open(f"{self.logs_path}/client_{self.client_id}_tempo.csv", mode="w", newline="") as tf:
                    writer = csv.writer(tf)
                    writer.writerow(["total_time_seconds"])
                    writer.writerow([total_time])
                self.log_file.close()
                self.model.debug_print_tree()
                return parameters, 1, {
                    "accuracy": self.safe_metric_get(self.accuracy),
                    "f1":       self.safe_metric_get(self.f1),
                    "continue_training": False,
                }
            #self.replay_buffer.append((x, y))
            self.last_instance = (x, y)
            self.model.last_instance = self.last_instance
            y_pred = self.model.predict_one(x)
            self.model.learn_one(x, y)
            self.accuracy.update(y, y_pred)
            self.f1.update(y, y_pred)

            # Current instance log
            self.log_writer.writerow({
                "instances": self.count_instances,
                "round": self.round,
                "splits": self.model.n_branches,
                "depth": self.model.depth(),
                "nodes": self.model.n_nodes,
                "leaves": self.model.n_leaves,
                "accuracy": self.safe_metric_get(self.accuracy),
                "f1": self.safe_metric_get(self.f1),
            })
            self.log_file.flush()
            self.count_instances += 1

            # If we have a pending split, we send stats to the server.
            if self.model._pending_split is not None:
                self.model._frozen = True  # Keeps locked until the global split is applied.
                self.frozen_since_round = self.round

                stats_msg = json.dumps(self.send_tree_stats())
                print("📤 Sending stats of detected split.")
                #print("++++++ AFTER ++++++++")
                #self.model.debug_print_tree()
                self.round += 1
                return parameters, 1, {
                    "tree_stats": stats_msg,
                    "accuracy": self.safe_metric_get(self.accuracy),
                    "f1": self.safe_metric_get(self.f1),
                    "continue_training": True,
                }


from pathlib import Path
def start_client(client_id):
    """Starts a Flower client and captures connection errors."""
    n_clients = 10
    abs_path = Path("CIC_IOT") / "nodes" / f"{n_clients}nodes"
    file_path = abs_path / f"client_{client_id}.csv"
    path_logs = Path("logs") / abs_path
    os.makedirs(path_logs, exist_ok=True)
    client = FedVFDTClient(file_path, path_logs, client_id)
    MAX_RETRIES = 1
    try:
        for attempt in range(MAX_RETRIES):
            try:
                fl.client.start_client(
                    #server_address="server:8083", for docker
                    server_address="127.0.0.1:8083",
                    client=client.to_client()
                )
                break  # Exits the loop if the connection is successful.
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    print(f"⚠️ Attempt {attempt+1}/{MAX_RETRIES} failed: Server unavailable.")
                    if attempt == MAX_RETRIES - 1:
                        print(f"✅ Client {client_id} terminated because the server shut down.")
                        return  # Exits the function without error, avoiding sys.exit(1)
                else:
                    print(f"❌ Unexpected error in the Client {client_id}: {e}")
                    return  # Prevents exiting with a critical error
    finally:
        client.close()

if __name__ == "__main__":
    client_id = int(sys.argv[1])
    start_client(client_id)