import copy
import uuid
from river.tree import HoeffdingTreeClassifier
from river.tree.nodes.branch import NumericBinaryBranch, NominalBinaryBranch
from river.tree import nodes as tree_nodes
from river.tree.nodes.leaf import HTLeaf
import pickle
import base64
from river.tree.nodes.branch import Branch
from river.tree.splitter import GaussianSplitter
from river import stats

class FederatedHoeffdingTree(HoeffdingTreeClassifier):
    """Adaptation of River's HoeffdingTreeClassifier for federated splits"""

    def __init__(self, grace_period=100, delta=1e-5, splitter=None, split_criterion = "gini"):
        super().__init__(grace_period=grace_period, delta=delta, splitter=splitter, split_criterion=split_criterion)
        self._pending_split = None
        self._frozen = False
        self._leaf_counters = {}
        self._instance_buffer = {}

    def depth(self):
        return self._get_depth(self._root)

    def _get_depth(self, node):
        if isinstance(node, HTLeaf):
            return 0
        elif isinstance(node, Branch):
            return 1 + max(self._get_depth(child) for child in node.children)
        else:
            return 0


    def learn_one(self, x, y, *, w=1.0):
        self.classes.add(y)
        self._train_weight_seen_by_model += w

        # 1) Initializes the root if necessary
        if self._root is None:
            self._root = self._new_leaf()
            # self.n_active_leaves = 1

        # 2) Finds the corresponding leaf
        leaf = self._get_leaf_for_instance(self._root, x)
        if leaf is None:
            return

        # 3) Ensures UUID
        if not hasattr(leaf, "uuid") or leaf.uuid is None:
            leaf.uuid = str(uuid.uuid4())
        uid = leaf.uuid

        # 4) Updates leaf statistics
        leaf.learn_one(x, y, tree=self)

        # 4.5) Stores the instance in the buffer for post-split reprocessing
        self._instance_buffer.setdefault(uid, []).append((x, y, w))

        # 5) Checks if the node is active
        if not leaf.is_active():
            return

        # 6) Updates counters
        self._leaf_counters.setdefault(uid, 0)
        self._leaf_counters[uid] += 1

        # 7) Obtains progress since the last split attempt
        weight_seen = getattr(leaf, "total_weight", 0)
        last_attempt = getattr(leaf, "last_split_attempt_at", 0)
        if (weight_seen - last_attempt) < self.grace_period:
            return  # Grace period not yet reached


        # 8) Split suggestions
        split_criterion = self._new_split_criterion()
        suggestions = leaf.best_split_suggestions(split_criterion, self)
        suggestions = [s for s in suggestions if s.merit != float("-inf")]
        if not suggestions:
            leaf.last_split_attempt_at = weight_seen
            self._leaf_counters[uid] = 0
            self._pending_split = None
            self._frozen = False
            return
        suggestions.sort()

        best = suggestions[-1]
        second = suggestions[-2] if len(suggestions) > 1 else None

        epsilon = self._hoeffding_bound(
            split_criterion.range_of_merit(leaf.stats),
            self.delta,
            leaf.total_weight,
            )

        if second:
            merit_diff = best.merit - second.merit
            if merit_diff > epsilon or epsilon < self.tau:
                print(f"✅ [Guaranteed split] Difference > epsilon or epsilon < tau")
            else:
                leaf.last_split_attempt_at = weight_seen
                self._leaf_counters[uid] = 0
                self._pending_split = None # controls that the client does not send repeated splits
                self._frozen = False
                return

        # 9) Split preparation
        feature   = getattr(best, "attribute", None) or getattr(best, "feature", None)
        threshold = getattr(best, "threshold", None) or getattr(best, "split_info", None)
        self._pending_split = {
            "leaf_uuid":   uid,
            "feature":     feature,
            "threshold":   threshold,   # For binary splits
            "split_info":  getattr(best, "split_info", None),  # For multinomial splits
            "stats":       copy.deepcopy(leaf.stats),
            "gini":        best.merit,
            "epsilon":     epsilon,
            "split_test":  best,
            "depth":       getattr(leaf, "depth", None),
            "leaf_path":   self._get_path_to_node(self._root, leaf),
            "suggestions": suggestions,
        }

        # 10) Freezes local learning and marks the split attempt
        self._frozen = True
        leaf.last_split_attempt_at = weight_seen
        self._leaf_counters[uid] = 0
        print(f"📤 Stored split proposal: feature={self._pending_split['feature']}, merit={best.merit:.4f}")
        print(f"⏸️ Learning frozen: waiting for global split on leaf {uid}")

        # 11) Stores the last instance globally (for post-split reapplication)
        self.last_instance = (x, y)

    def _get_all_active_leaves(self):
        leaves = []
        def traverse(node):
            if isinstance(node, HTLeaf) and node.is_active():
                leaves.append(node)
            elif hasattr(node, "children"):
                children = node.children
                if isinstance(children, dict):
                    for child in children.values():
                        traverse(child)
                elif isinstance(children, list):
                    for child in children:
                        traverse(child)

        traverse(self._root)
        return leaves


    def apply_aggregated_split(self, decision: dict):
        """Applies the split received from the server using path as a universal reference."""

        feature   = decision.get("feature")
        threshold = decision.get("threshold")
        stats     = decision.get("node_stats", decision.get("stats"))
        value     = decision.get("value", None)
        split_ser = decision.get("split_test", None)
        path      = decision.get("path")
        children_stats = decision.get("children_stats")

        if feature is None or stats is None or path is None:
            print("⚠️ Invalid split; aborting (missing feature/stats/path).")
            return

        # Finds the target node using the path
        target = self._find_node_by_path(self._root, path)
        if target is None:
            print(f"⚠️ Could not find leaf at path={path}; aborting split.")
            print("🌳 Client's current tree:")
            self.debug_print_tree()
            return

        # Forces the presence of the feature in the local splitter, if necessary
        if hasattr(target, "splitter") and isinstance(target.splitter, GaussianSplitter):
            fs = getattr(target.splitter, "_feature_stats", None)
            if fs is not None and feature not in fs:
                print(f"🩹 Manually adding feature '{feature}' to GaussianSplitter.")
                fs[feature] = stats.Var()

        if self._pending_split is None or "suggestions" not in self._pending_split:
            print(f"⚠️ Client has no pending_split or local suggestions. Ignoring global split.")
            selected_local_suggestion = None  # Força uso do split global
        else:
            suggestions = self._pending_split.get("suggestions")
            selected_local_suggestion = None
            for suggestion in suggestions:
                local_feat = getattr(suggestion, "feature", None)
                if str(local_feat) == str(feature):
                    selected_local_suggestion = suggestion
                    break

        if self._pending_split is not None:
            # Find the local suggestion that matches the feature from the global split
            suggestions = self._pending_split.get("suggestions")
            selected_local_suggestion = None
            for suggestion in suggestions:
                local_feat = getattr(suggestion, "feature", None)
                if str(local_feat) == str(feature):
                    selected_local_suggestion = suggestion
                    break
            if selected_local_suggestion is None:
                print(f"⚠️ Client did not propose feature '{feature}' — applying global split anyway.")
                local_threshold = decision.get("threshold")  # global threshold
                split_test_ser = decision.get("split_test")
                children_stats_ser = decision.get("children_stats")
                if not split_test_ser or not children_stats_ser:
                    print("❌ Incomplete global split — missing split_test or children_stats.")
                    return
                try:
                    split_test = pickle.loads(base64.b64decode(split_test_ser.encode("utf-8")))
                except Exception as e:
                    print(f"❌ Error while deserializing split_test: {e}")
                    return
                try:
                    local_children_stats = pickle.loads(base64.b64decode(children_stats_ser.encode("utf-8")))
                except Exception as e:
                    print(f"❌ Error while deserializing children_stats: {e}")
                    return
                local_feature_type = isinstance(split_test, GaussianSplitter)
            else:
                local_threshold = selected_local_suggestion.split_info
                #split_test = selected_local_suggestion.split_test
                local_children_stats = selected_local_suggestion.children_stats
                local_feature_type = selected_local_suggestion.numerical_feature

            global_feature = feature


            # Creates two children
            left = self._new_leaf(initial_stats={}, parent=target)
            right = self._new_leaf(initial_stats={}, parent=target)

            # Forces unique IDs if they do not exist
            for node in [left, right]:
                if not getattr(node, 'uuid', None):
                    node.uuid = str(uuid.uuid4())

            split_test_obj = pickle.loads(base64.b64decode(split_ser))

            # Branch
            if local_feature_type: # local_feature_type = true
                # Split numérico
                branch = NumericBinaryBranch(
                    feature=global_feature,
                    threshold=local_threshold,
                    stats=copy.deepcopy(target.stats),
                    depth=target.depth,
                    left=left,
                    right=right,
                )
            if not local_feature_type:
                branch = NominalBinaryBranch(
                    feature=global_feature,
                    value=local_threshold,
                    #split_test=selected_local_suggestion,
                    split_test=split_test_obj,
                    stats=copy.deepcopy(target.stats),
                    depth=target.depth,
                    left=left,
                    right=right,
                )

            # Keeps the original UUID of the leaf (optional, for debug/consistency)
            branch.uuid = getattr(target, "uuid", None)

            # Replaces in the correct position
            if target is self._root:
                print("🌳 Replacing root directly (empty path or root).")
                self._root = branch
            else:
                if not self._replace_leaf_with_branch(self._root, target, branch):
                    print("❌ Failed to replace leaf at the specified path.")

            # Clears temporary states
            self._pending_split = None
            self._frozen = False

            # Reapplies the last real instance after the split, if available
            if hasattr(self, "last_instance") and self.last_instance:
                print(f"🔁 Reapplying instance after split: {self.last_instance[0]}")
                self.learn_one(*self.last_instance)

            split_desc = f"{feature} {'≤ ' + str(threshold) if threshold is not None else '== ' + str(value)}"
            print(f"✅ Global split applied at path={path}: {split_desc}")

    def _get_path_to_node(self, node, target, path=None):
        if path is None:
            path = []
        if node is target:
            return path
        if hasattr(node, "children"):
            children = node.children
            if len(children) == 2:
                left, right = children
                left_path = self._get_path_to_node(left, target, path + ["left"])
                if left_path is not None:
                    return left_path
                right_path = self._get_path_to_node(right, target, path + ["right"])
                if right_path is not None:
                    return right_path
        return None

    def _find_node_by_path(self, node, path):
        current = node
        for step in path:
            if not hasattr(current, "children") or len(current.children) != 2:
                return None
            if step == "left":
                current = current.children[0]
            elif step == "right":
                current = current.children[1]
            else:
                return None
        return current


    def _replace_leaf_with_branch(self, node, target, branch):
        if node is target:
            return True
        if hasattr(node, "children"):
            for i, c in enumerate(node.children):
                if c is target:
                    node.children[i] = branch
                    return True
                if self._replace_leaf_with_branch(c, target, branch):
                    return True
        return False

    def _find_node_by_uuid(self, node, uid):
        if getattr(node, "uuid", None) == uid:
            return node
        if hasattr(node, "children"):
            for c in node.children:
                found = self._find_node_by_uuid(c, uid)
                if found:
                    return found
        return None

    def _get_leaf_for_instance(self, node, x):
        """Traverses the tree until it finds the leaf where x falls."""
        if node is None:
            return None

        # If it is a leaf (HTLeaf) or marked as a leaf
        if isinstance(node, HTLeaf) or getattr(node, "_is_leaf", False):
            return node

        # List of children (can be dict or list)
        children = node.children if isinstance(node.children, list) else list(node.children.values())

        # Numeric binary branch
        if isinstance(node, NumericBinaryBranch):
            feat, thr = node.feature, node.threshold
            if x.get(feat, 0) <= thr:
                return self._get_leaf_for_instance(children[0], x)
            else:
                return self._get_leaf_for_instance(children[1], x)

        # Nominal binary branch
        if isinstance(node, NominalBinaryBranch):
            feat, val = node.feature, node.value
            if x.get(feat) == val:
                return self._get_leaf_for_instance(children[0], x)
            else:
                return self._get_leaf_for_instance(children[1], x)

        # Generic fallback (in case there are other types of branches)
        for child in children:
            leaf = self._get_leaf_for_instance(child, x)
            if leaf is not None:
                return leaf
        return None

    def debug_print_tree(self):
        """Prints the decision tree correctly distinguishing leaves and different types of branches."""
        def recurse(node, depth=0):
            pad = "    " * depth
            nid = getattr(node, "uuid", "<no-id>")

            # Leaf (HTLeaf or any node without children)
            if getattr(node, "_is_leaf", False) or not hasattr(node, "children"):
                stats = getattr(node, "stats", {})
                total_weight = getattr(node, "total_weight", 0)
                print(f"{pad}🌿 Leaf(id={nid}) stats={stats} total_weight={total_weight}")

            # Numerical Branch
            elif isinstance(node, NumericBinaryBranch):
                feat = node.feature
                thr  = node.threshold
                stats = getattr(node, "stats", {})
                print(f"{pad}🔹 NumBranch(id={nid}) {feat} ≤ {thr} stats={stats}")
                # children in .children[0] and .children[1]
                children = node.children
                recurse(children[0], depth + 1)
                recurse(children[1], depth + 1)

            # Nominal binary branch
            elif isinstance(node, NominalBinaryBranch):
                feat = node.feature
                # Attempts to extract the split value, if it exists
                val = getattr(node.split_test, "value", "<None>")
                stats = getattr(node, "stats", {})
                print(f"{pad}🔹 NomBranch(id={nid}) {feat} == {val} stats={stats}")
                children = node.children
                recurse(children[0], depth + 1)
                recurse(children[1], depth + 1)

            # Any other node (for safety)
            else:
                print(f"{pad}? Node(id={nid}) unknown type: {type(node)} stats={getattr(node, 'stats', {})}")
                for child in getattr(node, "children", []):
                    recurse(child, depth + 1)

        print("=== Decision tree ===")
        if self._root is None:
            print("🚫 Uninitialized tree.")
        else:
            recurse(self._root, 0)
        print("=========================")