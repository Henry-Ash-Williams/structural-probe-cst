from dataclasses import dataclass
from functools import cached_property
from typing import Self

import numpy as np
from tree_sitter import Node

__all__ = ["Cst", "CstNode"]


def preorder_traversal(tree):
    cursor = tree.walk()

    nodes = {}
    next_id = 0
    path = []

    root_id = next_id
    next_id += 1
    nodes[root_id] = {"parent": None, "children": [], "node": cursor.node}

    path.append(root_id)

    while True:
        current_id = path[-1]

        if cursor.goto_first_child():
            child_id = next_id
            next_id += 1
            nodes[current_id]["children"].append(child_id)
            nodes[child_id] = {
                "parent": current_id,
                "children": [],
                "node": cursor.node,
            }
            path.append(child_id)
        else:
            while True:
                if len(path) == 1:
                    return nodes

                if cursor.goto_next_sibling():
                    path.pop()
                    sibling_id = next_id
                    next_id += 1
                    parent_id = path[-1]
                    nodes[parent_id]["children"].append(sibling_id)
                    nodes[sibling_id] = {
                        "parent": parent_id,
                        "children": [],
                        "node": cursor.node,
                    }
                    path.append(sibling_id)
                    break
                else:
                    cursor.goto_parent()
                    path.pop()


@dataclass
class CstNode:
    text: str | None
    id: int
    parent: int | None
    children: list[int]
    start: int
    end: int

    @classmethod
    def from_dict(cls, dict) -> Self:
        return cls(
            text=dict["text"],
            id=dict["id"],
            parent=dict["parent"],
            children=dict["children"],
            start=dict["start"],
            end=dict["end"],
        )

    def to_dict(self):
        return {
            "text": self.text,
            "id": self.id,
            "parent": self.parent,
            "children": self.children,
            "start": self.start,
            "end": self.end,
        }

    def is_leaf(self):
        return self.children == []

    def is_root(self):
        return self.parent is None


class Cst:
    def __init__(self, nodes, source, language):
        self.nodes = [None] * len(nodes)
        self.source = source.decode("utf-8")
        self.language = language

        for id, node in nodes.items():
            start, end = node["node"].byte_range

            self.nodes[id] = CstNode(
                id=id,
                parent=node["parent"],
                children=node["children"],
                text=node["node"].text.decode("utf-8")
                if node["children"] == []
                else None,
                start=start,
                end=end,
            )

    @classmethod
    def from_parse_tree(cls, language: str, source: bytes, parse_tree: Node) -> Self:
        nodes = preorder_traversal(parse_tree)

        return cls(language=language, nodes=nodes, source=source)

    @classmethod
    def from_dict(cls, dict) -> Self:
        tmp = cls(
            language=dict["language"], source=bytes(dict["source"], "utf-8"), nodes={}
        )

        tmp.nodes = [CstNode.from_dict(node) for node in dict["nodes"]]
        return tmp

    def to_dict(self):
        return {
            "language": self.language,
            "source": self.source,
            "nodes": [node.to_dict() for node in self.nodes],
        }

    def __path_to_root(self, node: CstNode) -> list[int]:
        """
        Finds the path from `node` to the root of the tree

        Returns a list of node IDs
        """
        path = [node.id]

        while node.parent is not None:
            node = self.nodes[node.parent]
            path.append(node.id)

        return path

    def __compute_depth(self, node: CstNode):
        """
        Computes the depth of `node` within the tree
        """
        return len(self.__path_to_root(node)) - 1

    def __compute_distance(self, source: CstNode, target: CstNode):
        """
        Computes the length of the path between the source and target nodes
        """
        lca = max(set(self.__path_to_root(source)) & set(self.__path_to_root(target)))
        lca_depth = self.depth[lca]
        return self.depth[source.id] + self.depth[target.id] - 2 * lca_depth

    @cached_property
    def root_id(self) -> CstNode:
        potential_root_nodes = [node for node in self.nodes if node.is_root()]

        assert len(potential_root_nodes) > 0, (
            "Expected exactly one root node, found none"
        )
        assert len(potential_root_nodes) < 2, (
            "Expected exactly one root node, found multiple"
        )

        return potential_root_nodes[0]

    @cached_property
    def leaf_indicies(self) -> np.ndarray:
        return np.array([node.id for node in self.nodes if node.is_leaf()])

    @cached_property
    def leaves(self) -> list[CstNode]:
        """
        A list of all leaf nodes
        """
        return [self.nodes[leaf_id] for leaf_id in self.leaf_indicies]

    @cached_property
    def num_leaves(self) -> int:
        """
        Number of leaves in the tree
        """
        return len(self.leaf_indicies)

    @cached_property
    def distance(self) -> np.ndarray:
        """
        A symmetrical NxN matrix containing the path distance between nodes `i` and `j`
        """
        distance_matrix = np.zeros((self.num_leaves, self.num_leaves), dtype=np.uint)

        for i, leaf_i in enumerate(self.leaves):
            for j, leaf_j in enumerate(self.leaves):
                distance_matrix[i, j] = distance_matrix[j, i] = self.__compute_distance(
                    leaf_i, leaf_j
                )

        return distance_matrix

    @cached_property
    def depth(self) -> np.ndarray:
        """
        A list storing the depth of each node
        """
        return np.array([self.__compute_depth(node) for node in self.nodes])
