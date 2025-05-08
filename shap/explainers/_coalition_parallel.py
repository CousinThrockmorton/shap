import math
from itertools import chain, combinations, product

import numpy as np  # numpy base
from numba import njit  # just in time compiler

from .. import links  # shap modules
from ..explainers._explainer import Explainer
from ..models import Model
from ..utils import MaskedModel, make_masks, safe_isinstance


class CoalitionExplainer_p(Explainer):
    """A parallel implementation of the coalition-based explainer that uses Owen values to explain model predictions."""

    def __init__(
        self,
        model,
        masker,
        *,
        output_names=None,
        link=links.identity,
        linearize_link=True,
        feature_names=None,
        partition_tree=None,
        **call_args,
    ):
        """Initialize the parallel coalition explainer with a model and masker."""
        super().__init__(
            model,
            masker,
            link=link,
            linearize_link=linearize_link,
            algorithm="partition",
            output_names=output_names,
            feature_names=feature_names,
        )
        self.input_shape = masker.shape[1:] if hasattr(masker, "shape") and not callable(masker.shape) else None
        if not safe_isinstance(self.model, "shap.models.Model"):
            self.model = Model(self.model)

        self.expected_value = None
        self._curr_base_value = None

        if self.input_shape is not None and len(self.input_shape) > 1:
            self._reshaped_model = lambda x: self.model(x.reshape(x.shape[0], *self.input_shape))
        else:
            self._reshaped_model = self.model

        self.partition_tree = partition_tree

        if not callable(self.masker.clustering):
            self._clustering = self.masker.clustering
            self._mask_matrix = make_masks(self._clustering)

        if len(call_args) > 0:

            class CoalitionExplainer_p(self.__class__):
                def __call__(
                    self,
                    *args,
                    max_evals=500,
                    fixed_context=None,
                    main_effects=False,
                    error_bounds=False,
                    batch_size="auto",
                    outputs=None,
                    silent=False,
                ):
                    return super().__call__(
                        *args,
                        max_evals=max_evals,
                        fixed_context=fixed_context,
                        main_effects=main_effects,
                        error_bounds=error_bounds,
                        batch_size=batch_size,
                        outputs=outputs,
                        silent=silent,
                    )

            CoalitionExplainer_p.__call__.__doc__ = self.__class__.__call__.__doc__
            self.__class__ = CoalitionExplainer_p
            for k, v in call_args.items():
                self.__call__.__kwdefaults__[k] = v

    def __call__(
        self,
        *args,
        max_evals=500,
        fixed_context=None,
        main_effects=False,
        error_bounds=False,
        batch_size="auto",
        outputs=None,
        silent=False,
    ):
        return super().__call__(
            *args,
            max_evals=max_evals,
            fixed_context=fixed_context,
            main_effects=main_effects,
            error_bounds=error_bounds,
            batch_size=batch_size,
            outputs=outputs,
            silent=silent,
        )

    def explain_row(
        self,
        *row_args,
        max_evals,
        main_effects,
        error_bounds,
        batch_size,
        outputs,
        silent,
        fixed_context="auto",
    ):
        if fixed_context == "auto":
            fixed_context = None
        elif fixed_context not in [0, 1, None]:
            raise ValueError(f"Unknown fixed_context value passed (must be 0, 1 or None): {fixed_context}")

        fm = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *row_args)
        M = len(fm)
        m00 = np.zeros(M, dtype=bool)

        # if not fixed background or no base value assigned then compute base value for a row
        if self._curr_base_value is None or not getattr(self.masker, "fixed_background", False):
            base_output = fm(m00.reshape(1, -1), zero_index=0)[0]
            self._curr_base_value = np.array(base_output) if isinstance(base_output, (list, tuple)) else base_output

        # Handle multi-output predictions
        if isinstance(self._curr_base_value, np.ndarray) and self._curr_base_value.ndim > 0:
            num_outputs = len(self._curr_base_value)
            shap_values = np.zeros((M, num_outputs))
        else:
            num_outputs = 1
            shap_values = np.zeros(M)

        # build the hierarchy
        self.root = Node("Root")
        build_tree(self.partition_tree, self.root)
        self.combinations_list = generate_paths_and_combinations(self.root)
        
        self.masks, self.keys = create_masks1(self.root, self.masker.feature_names)
        self.masks_dict = dict(zip(self.keys, self.masks))
        
        self.mask_permutations = create_combined_masks(self.combinations_list, self.masks_dict)
        self.masks_list = [mask for _, mask, _ in self.mask_permutations]
        self.unique_masks_set = set(map(tuple, self.masks_list))
        self.unique_masks = [np.array(mask) for mask in self.unique_masks_set]

        # Step 2: Compute model results for all unique masks
        mask_results = {}
        for mask in self.unique_masks:
            result = fm(mask.reshape(1, -1))
            mask_results[tuple(mask)] = np.array(result) if isinstance(result, (list, tuple)) else result

        # Step 3: Compute marginals for permutations
        last_key_to_off_indexes, last_key_to_on_indexes, weights = map_combinations_to_unique_masks(
            self.mask_permutations, self.unique_masks
        )

        feature_name_to_index = {name: idx for idx, name in enumerate(self.masker.feature_names)}

        # Step 4: Implement Owen values weighting
        for last_key in last_key_to_off_indexes:
            off_indexes = last_key_to_off_indexes[last_key]
            on_indexes = last_key_to_on_indexes[last_key]
            weight_list = weights[last_key]

            for off_index, on_index, weight in zip(off_indexes, on_indexes, weight_list):
                off_result = mask_results[tuple(self.unique_masks[off_index])]
                on_result = mask_results[tuple(self.unique_masks[on_index])]

                if num_outputs > 1:
                    for i in range(num_outputs):
                        marginal_contribution = float((on_result[i] - off_result[i]) * weight)
                        shap_values[feature_name_to_index[last_key], i] += marginal_contribution
                else:
                    marginal_contribution = float((on_result - off_result) * weight)
                    shap_values[feature_name_to_index[last_key]] += marginal_contribution

        # Step 5: Return results
        return {
            "values": shap_values.copy(),
            "expected_values": self._curr_base_value,
            "mask_shapes": [s + () for s in fm.mask_shapes],
            "main_effects": None,
            "hierarchical_values": shap_values,
            "clustering": None,
            "output_indices": outputs,
            "output_names": getattr(self.model, "output_names", None),
        }

    def __str__(self):
        return "shap.explainers.PartitionExplainer()"


@njit
def lower_credit(i, value, M, values, clustering):
    if i < M:
        values[i] += value
        return
    li = int(clustering[i - M, 0])
    ri = int(clustering[i - M, 1])
    group_size = int(clustering[i - M, 3])
    lsize = int(clustering[li - M, 3]) if li >= M else 1
    rsize = int(clustering[ri - M, 3]) if ri >= M else 1
    assert lsize + rsize == group_size
    values[i] += value
    lower_credit(li, values[i] * lsize / group_size, M, values, clustering)
    lower_credit(ri, values[i] * rsize / group_size, M, values, clustering)


################################## HELPER FUNCTIONS ################
class Node:
    def __init__(self, key):
        self.key = key
        self.child = []
        self.permutations = []  # this may not be the greatest idea??
        self.weights = []

    def __repr__(self):
        return f"({self.key}): {self.child} -> {self.permutations} \\ {self.weights}"


def build_tree(d, root):
    if isinstance(d, dict):
        for key, value in d.items():
            node = Node(key)
            root.child.append(node)
            build_tree(value, node)
    elif isinstance(d, list):
        for item in d:
            node = Node(item)
            root.child.append(node)
    # get all the sibling permutations
    generate_permutations(root)


def compute_weight(total, selected):
    return 1 / (total * math.comb(total - 1, selected))


def all_subsets(iterable):
    return chain.from_iterable(combinations(iterable, n) for n in range(len(iterable) + 1))


def combine_masks(masks):
    combined_mask = np.logical_or.reduce(masks)
    return combined_mask


def create_partition_hierarchy(linkage_matrix, columns):
    def build_hierarchy(node, linkage_matrix, columns):
        if node < len(columns):
            return {columns[node]: columns[node]}
        else:
            left_child = int(linkage_matrix[node - len(columns), 0])
            right_child = int(linkage_matrix[node - len(columns), 1])
            left_subtree = build_hierarchy(left_child, linkage_matrix, columns)
            right_subtree = build_hierarchy(right_child, linkage_matrix, columns)
            return {f"cluster_{node}": {**left_subtree, **right_subtree}}

    root_node = len(linkage_matrix) + len(columns) - 1
    hierarchy = build_hierarchy(root_node, linkage_matrix, columns)
    return hierarchy[f"cluster_{root_node}"]


##########################################################################


def generate_permutations(node):
    if not node.child:  # Leaf node
        node.permutations = []
        return

    children_keys = [child.key for child in node.child]
    node.permutations = {}

    for i, child in enumerate(node.child):
        excluded = children_keys[:i] + children_keys[i + 1 :]
        generate_permutations(child)
        child.permutations = list(all_subsets(excluded))
        child.weights = [compute_weight(len(children_keys), len(permutation)) for permutation in child.permutations]


def get_all_leaf_values(node):
    leaves = []
    if not node.child:
        leaves.append(node.key)
    else:
        for child in node.child:
            leaves.extend(get_all_leaf_values(child))
    return leaves


def create_masks1(node, columns):
    masks = [np.zeros(len(columns), dtype=bool)]
    keys = [()]

    if not node.child:
        mask = columns == node.key
        masks.append(mask)
        keys.append(node.key)
    else:
        current_node_mask = columns.isin(get_all_leaf_values(node))
        masks.append(current_node_mask)
        keys.append(node.key)

        for subset in node.child:
            child_masks, child_keys = create_masks1(subset, columns)
            masks.extend(child_masks)
            keys.extend(child_keys)

    return masks, keys


def generate_paths_and_combinations(node):
    paths = []

    def dfs(current_node, current_path):
        current_path.append((current_node.key, current_node.permutations, current_node.weights))

        if not current_node.child:  # Leaf node
            paths.append(current_path[:])
        else:
            for child in current_node.child:
                dfs(child, current_path)

        current_path.pop()  # Backtrack

    dfs(node, [])

    combinations_list = []

    for path in paths:
        filtered_path = [(key, perms, weight) for key, perms, weight in path if perms]
        if filtered_path:
            node_keys, permutations, weights = zip(*filtered_path)
            path_combinations = list(product(*permutations))
            weight_combinations = list(product(*weights))
            weight_products = [np.prod(weight_tuple) for weight_tuple in weight_combinations]
            last_key = node_keys[-1]
            for i, combination in enumerate(path_combinations):
                combinations_list.append((last_key, combination, weight_products[i]))

    return combinations_list


def create_combined_masks(combinations, masks_dict):
    combined_masks = []
    for last_key, combination, weights in combinations:
        masks = []
        for keys in combination:
            if isinstance(keys, tuple) and not keys:
                continue
            for key in keys:
                if key in masks_dict:
                    masks.append(masks_dict[key])

        if masks:
            combined_mask = combine_masks(masks)
            combined_masks.append((last_key, combined_mask, weights))

            if last_key in masks_dict:
                combined_mask_with_last_key = combine_masks(masks + [masks_dict[last_key]])
                combined_masks.append((last_key, combined_mask_with_last_key, weights))
        else:
            combined_mask = np.zeros_like(list(masks_dict.values())[0])
            combined_masks.append((last_key, combined_mask, weights))

            if last_key in masks_dict:
                combined_mask_with_last_key = combine_masks([combined_mask, masks_dict[last_key]])
                combined_masks.append((last_key, combined_mask_with_last_key, weights))
    return combined_masks


def map_combinations_to_unique_masks(combined_masks, unique_masks):
    unique_mask_index_map = {tuple(mask): idx for idx, mask in enumerate(unique_masks)}
    last_key_to_off_indexes = {}
    last_key_to_on_indexes = {}
    weights = {}

    for i, (last_key, combined_mask, weight) in enumerate(combined_masks):
        mask_tuple = tuple(combined_mask)
        unique_index = unique_mask_index_map[mask_tuple]

        if i % 2 == 0:  # Even index -> OFF value
            if last_key not in last_key_to_off_indexes:
                last_key_to_off_indexes[last_key] = []
                weights[last_key] = []
            last_key_to_off_indexes[last_key].append(unique_index)
            weights[last_key].append(weight)
        else:  # Odd index -> ON value
            if last_key not in last_key_to_on_indexes:
                last_key_to_on_indexes[last_key] = []
            last_key_to_on_indexes[last_key].append(unique_index)

    return last_key_to_off_indexes, last_key_to_on_indexes, weights
