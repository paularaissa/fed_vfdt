import flwr as fl
import json
import os
import csv
import math
from my_server import MyServer
from flwr.common import (
    FitIns,
    Parameters,
)
from flwr.server.strategy import FedAvg
from datetime import datetime
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

class StrategyVFDT(FedAvg):
    """
    Initializes the custom federated learning strategy designed for training Very Fast Decision Trees (VFDT)
    in a distributed setting.

    Parameters
    ----------
    min_available_clients : int
        The minimum number of clients that must be available before a training round can be initiated.

    min_fit_clients : int
        The minimum number of clients required to participate in the training phase of each round.

    min_evaluate_clients : int
        The minimum number of clients required to participate in the evaluation phase of each round.

    path_logs : str
        The directory path where server-side logs are stored, including round-level metrics and global split decisions.

    global_grace_period : int
        The total grace period, used to compute each client's local grace period based on the number of participating clients.
        This ensures that the total number of instances processed across clients remains approximately balanced.

    aggregation_strategy : str, optional (default="quorum")
        The strategy used to aggregate split proposals from clients and decide on the global split.
        Supported options are:

        - `"quorum"`: Selects the most frequently proposed feature, but only if it meets a minimum support threshold (see `support_percent`).
          If no feature satisfies the quorum, no split is applied.

        - `"best-merit"`: Selects the feature with the highest merit score (e.g., Gini index), regardless of how many clients proposed it.

        - `"majority-vote"`: Selects the feature proposed by the majority of clients. In case of a tie, the feature with the highest merit among the tied ones is selected.

    support_percent : int, optional (default=60)
        The minimum percentage of clients that must propose the same feature in order for it to be considered valid under the `"quorum"` strategy.
        This parameter is ignored when using the `"best-merit"` or `"majority-vote"` strategies.
    """
    def __init__(self, min_available_clients, min_fit_clients, min_evaluate_clients,
                 path_logs, global_grace_period,
                 aggregation_strategy="quorum", support_percent=60):
        super().__init__()
        self.received_reply = False  # Flag to check whether a response has already been received

        self.increase_num_rounds_by = 0  # Initially, no additional rounds are added
        self.received_first_result = False  # Flag indicating whether a result has already been received
        self.first_client_result = None  # Stores the first result

        self.min_available_clients = min_available_clients
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients

        self.path_logs = path_logs
        self.global_split_info = {}  # Dictionary to store the best feature and aggregated values
        self.global_grace_period = global_grace_period

        self.aggregation_strategy = aggregation_strategy
        self.support_percent = support_percent

        self.aggregated_accuracy = 0
        self.aggregated_f1 = 0

    def __repr__(self) -> str:
        return "FederatedHoeffdingTreeStrategy"

    def log_server_round(self, round_num, best_feature, global_stats, num_clients, accuracy, f1):
        os.makedirs(self.path_logs, exist_ok=True)
        log_path = f"{self.path_logs}server_log.csv"
        file_exists = os.path.isfile(log_path)

        row = {
            "round": round_num,
            "best_feature": best_feature,
            "gini_index": global_stats.get("gini_index"),
            "hoeffding_bound": global_stats.get("hoeffding_bound"),
            "accuracy": accuracy,
            "f1": f1,
            "num_clients": num_clients,
            "timestamp": datetime.now().isoformat(),
        }

        with open(log_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def aggregate_fit(self, server_round, results, failures):
        """Aggregates client results and determines whether training should continue."""
        print(f"📊 Round {server_round}: Receiving results from {len(results)} clients...")

        # If no client responded, do not perform aggregation
        if len(results) == 0:
            print("⚠️ No clients responded! Retaining previous parameters.")
            return None, {}

        # Receives the data sent by the clients
        client_tree_stats = []
        best_feature = None
        for client, res in results:
            try:
                tree_stats_json = res.metrics.get("tree_stats", "[]")  # Retrieving tree_stats from the client
                tree_stats = json.loads(tree_stats_json)  # Converting JSON to dictionary

                if tree_stats:
                    client_tree_stats.append(tree_stats)
                    print(f"📥 Client {client.cid} sent statistics: {tree_stats}")

            except json.JSONDecodeError:
                print(f"⚠️ Error decoding tree_stats from client {client.cid}")


        # Selects the best feature based on gain_local (gini - bound)
        best = None
        if client_tree_stats:
            if self.aggregation_strategy == "quorum":
                best = self._aggregate_with_quorum(client_tree_stats)
            elif self.aggregation_strategy == "best-merit":
                best = self._aggregate_best_merit(client_tree_stats)
            elif self.aggregation_strategy == "majority-vote":
                best = self._aggregate_majority_vote(client_tree_stats)
            else:
                raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
        if best:
            print(f"🏆 Best global feature selected: {best['feature']} (gini index: {best['gini_index']:.6f})")

            # Finds the dictionary with the highest gain_local
            # best = max(
            #     client_tree_stats,
            #     key=lambda d: d.get("gini_index", float("-inf"))
            # )
            # Parameter: minimum percentage of clients required to propose the same feature
            # Groups splits by feature
            # feature_groups = defaultdict(list)
            # for stats in client_tree_stats:
            #     key = stats["feature"]
            #     feature_groups[key].append(stats)

            # Evaluates groups with sufficient quorum
            # candidates = []
            # for feature, group in feature_groups.items():
            #     if len(group) >= min_clients_support:
            #         gini_sum = sum(g["gini_index"] for g in group)
            #         candidates.append((feature, group, gini_sum))

            # if candidates:
            #     # Selects the group with the highest accumulated Gini score
            #     best_feature, best_group, _ = max(candidates, key=lambda x: x[2])
            #     best = max(best_group, key=lambda d: d["gini_index"])  # Best local split in this group
            #
            #     best_feature        = best["feature"]
            #     best_threshold      = best["threshold"]
            #     best_gini           = best["gini_index"]
            #     best_hoeffding_bound= best["hoeffding_bound"]
            #     best_split_test     = best.get("split_test")
            #     best_leaf_id        = best["leaf_id"]      # ← Retrieves the leaf_id of the winner
            #     best_node_stats     = best["node_stats"]   # ← Retrieves the node_stats of the winner
            #     best_gain_global    = best["gain_local"]   # ← Defines gain_global
            #
            #     print(f"🏆 Best global feature chosen: {best_feature} (gini index: {best_gini:.6f})")

            # Updates the global split information
            self.global_split_info = {
                "leaf_id": best["leaf_id"],
                "path": best.get("path", []),
                "feature": best["feature"],
                "threshold": best["threshold"],
                "gini_index": best["gini_index"],
                "hoeffding_bound": best["hoeffding_bound"],
                "stats": best["node_stats"],
                "gain_global": best["gain_local"],
                "split_test": best.get("split_test"),
                "children_stats": best.get("children_stats"),
            }

            # Filters only the results that contain valid accuracy and F1 scores
            accuracy_list = [res.metrics["accuracy"] for _, res in results if "accuracy" in res.metrics]
            f1_list = [res.metrics["f1"] for _, res in results if "f1" in res.metrics]
            # If more than one client responded, aggregate the metrics
            if len(accuracy_list) > 0:
                self.aggregated_accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0.0
            if len(f1_list) > 0:
                self.aggregated_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
            print(f"📊 Aggregated accuracy: {self.aggregated_accuracy:.4f}")
            print(f"📊 Aggregated F1 score: {self.aggregated_f1:.4f}")
            self.log_server_round(
                round_num = server_round,
                best_feature = best["feature"],
                global_stats={
                    "gini_index": best["gini_index"],
                    "hoeffding_bound": best["hoeffding_bound"]
                },
                num_clients=len(results),
                accuracy=self.aggregated_accuracy,
                f1=self.aggregated_f1
            )
        else:
            print("⚠️ No split aggregated in this round.")
            self.global_split_info = {}  # Reset in case no aggregation is performed
            self.log_server_round(
                round_num=server_round,
                best_feature=None,
                global_stats={},
                num_clients=len(results),
                accuracy=self.aggregated_accuracy,
                f1=self.aggregated_f1
            )

        # Determines whether additional rounds will be executed
        if any(res.metrics.get("continue_training", False) for _, res in results):
            self.increase_num_rounds_by = 1  # Adds one more round
        else:
            print("✅ All clients have finished training. No new round will be started.")
            self.increase_num_rounds_by = 0  # Terminates if all clients have completed training

        return None, {}

    # Quorum-based strategy
    def _aggregate_with_quorum(self, splits):
        feature_groups = defaultdict(list)
        for stats in splits:
            feature_groups[stats["feature"]].append(stats)

        min_support = math.ceil(self.support_percent / 100 * self.min_fit_clients)
        candidates = [
            group for group in feature_groups.values() if len(group) >= min_support
        ]

        if not candidates:
            print(f"⚠️ No feature met the quorum threshold of {self.support_percent}%")
            return None

        best_group = max(candidates, key=lambda g: sum(s["gini_index"] for s in g))
        return max(best_group, key=lambda s: s["gini_index"])


    def _aggregate_best_merit(self, splits):
        # Checks if all clients have sent splits
        if len(splits) < self.min_fit_clients:
            print(f"⚠️ Split ignored: only {len(splits)}/{self.min_fit_clients} clients sent suggestions.")
            return None  # No split will be applied

        # Selects the split with the highest Gini Index
        best_split = max(splits, key=lambda s: s["gini_index"])
        print(f"✅ Split selected by highest merit: feature={best_split['feature']} | Gini={best_split['gini_index']:.5f}")
        return best_split


    def _aggregate_majority_vote(self, splits):
        from collections import defaultdict, Counter

        # Groups the proposals by leaf (using path as the key)
        groups_by_path = defaultdict(list)
        for s in splits:
            key = tuple(s["path"])
            groups_by_path[key].append(s)

        if not groups_by_path:
            print("⚠️ No splits to aggregate.")
            return None

        best_group = None
        best_votes = 0

        # In each leaf, votes for the most frequent feature
        for path, group in groups_by_path.items():
            features = [s["feature"] for s in group]
            counter = Counter(features)
            feature, votes = counter.most_common(1)[0]

            print(f"🗳️ Path {path} — voted feature: {feature} ({votes} votes)")

            if votes > best_votes:
                best_votes = votes
                best_group = [s for s in group if s["feature"] == feature]

            elif votes == best_votes:
                # Breaks ties by highest merit
                candidate = [s for s in group if s["feature"] == feature]
                current_best = max(best_group, key=lambda s: s["gini_index"])
                challenger = max(candidate, key=lambda s: s["gini_index"])
                if challenger["gini_index"] > current_best["gini_index"]:
                    best_group = candidate

        if best_group:
            # Returns the best proposal of the most voted feature within the selected leaf
            return max(best_group, key=lambda s: s["gini_index"])

        print("⚠️ No valid group passed majority-vote.")
        return None


    # Majority-vote strategy
    # def _aggregate_majority_vote(self, splits):
    #     from collections import Counter
    #     votes = Counter(s["feature"] for s in splits)
    #     top_count = votes.most_common(1)[0][1]
    #     top_features = [f for f, c in votes.items() if c == top_count]
    #
    #     if len(top_features) == 1:
    #         chosen = top_features[0]
    #     else:
    #         splits_tied = [s for s in splits if s["feature"] in top_features]
    #         chosen = max(splits_tied, key=lambda s: s["gini_index"])["feature"]
    #
    #     for s in splits:
    #         if s["feature"] == chosen:
    #             return s

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Configures the upcoming training rounds and sends the aggregation to clients"""
        config = {
            "global_split_info": json.dumps(self.global_split_info),  # Sends the aggregated data
            "trigger_update": True  # Forces the client to perform an extra learn_one after the data stream ends
        }

        # Validation: checks if global_split_info contains the required fields
        required_fields = ["leaf_id", "feature", "threshold", "stats", "split_test"]
        missing = [field for field in required_fields if field not in self.global_split_info]
        if missing:
            print(f"⚠️ Atenção: campos faltando na global_split_info: {missing}")

        # Samples the clients that will participate in the round
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        local_grace = max(1, int(self.global_grace_period / max(1, len(clients))))
        config["local_grace_period"] = local_grace

        fit_ins = FitIns(parameters, config)

        # Clears the split information after use
        self.global_split_info = {}

        return [(client, fit_ins) for client in clients]


import subprocess
def start_dashboard():
    dashboard_path = os.path.join("dashboard", "dashboard.py")
    subprocess.Popen(["python", dashboard_path])
    print("🟢 Dashboard started at http://localhost:8050")

def main():
    print("🔄 Creating strategy...")
    n_clients = 10
    global_grace_period = 200
    path_logs = f"logs/CIC_IOT/nodes/{n_clients}nodes/"
    os.makedirs(path_logs, exist_ok=True)
    strategy = StrategyVFDT(min_available_clients=n_clients,
                            min_fit_clients=n_clients,
                            min_evaluate_clients=n_clients,
                            path_logs=path_logs,
                            global_grace_period=global_grace_period,
                            #aggregation_strategy="best-merit",
                            aggregation_strategy="majority-vote",
                            #aggregation_strategy="quorum",
                            support_percent=60,
                            )

    print("🔄 Creating server...")
    server_new = MyServer(strategy=strategy)
    print("🚀 Starting the server on port 8083...")
    fl.server.start_server(
        server_address="0.0.0.0:8083",  # Ensures the server listens on port 8083
        server=server_new,
        config=fl.server.ServerConfig(num_rounds=1)
    )


if __name__ == "__main__":
    print("🔄 Starting server_app.py...")
    main()