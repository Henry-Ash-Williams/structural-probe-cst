import json
from dataclasses import dataclass
from functools import cached_property

import tree_sitter as ts
from datasets import load_dataset
from tqdm import tqdm
from tree_sitter import Query, QueryCursor
from tree_sitter_language_pack import (
    detect_language_from_extension,
    get_language,
    get_parser,
)

from data import Cst


@dataclass
class SupportedLanguage:
    ts_language: ts.Language
    ts_parser: ts.Parser
    query: str

    @cached_property
    def query_cursor(self):
        q = Query(self.ts_language, f"{self.query} @functions")
        return QueryCursor(q)


FUNCTIONS_PER_LANGUAGE = 1000


SUPPORTED_LANGUAGES = {
    "python": SupportedLanguage(
        get_language("python"), get_parser("python"), """(function_definition _)"""
    ),
    "c": SupportedLanguage(
        get_language("c"), get_parser("c"), """(function_definition _)"""
    ),
    "cpp": SupportedLanguage(
        get_language("cpp"), get_parser("cpp"), """(function_definition _)"""
    ),
    "java": SupportedLanguage(
        get_language("java"),
        get_parser("java"),
        """[
  (method_declaration _)
  (constructor_declaration _)
  (lambda_expression _)
]""",
    ),
    "javascript": SupportedLanguage(
        get_language("javascript"),
        get_parser("javascript"),
        """[
  (function_declaration _)
  (generator_function_declaration _)
  (function_expression _)
  (generator_function _)
  (arrow_function _)
  (method_definition _)
]""",
    ),
    "rust": SupportedLanguage(
        get_language("rust"),
        get_parser("rust"),
        """[
  (function_item _)
  (closure_expression _)
]""",
    ),
    "go": SupportedLanguage(
        get_language("go"),
        get_parser("go"),
        """[
  (function_declaration _)
  (method_declaration _)
  (func_literal _)
]""",
    ),
    "haskell": SupportedLanguage(
        get_language("haskell"),
        get_parser("haskell"),
        """[
  (function _)
  (bind _)
]""",
    ),
    "php": SupportedLanguage(
        get_language("php"),
        get_parser("php"),
        """[
  (function_definition _)
  (method_declaration _)
  (anonymous_function _)
  (arrow_function _)
]""",
    ),
}


def walk_tree(tree: ts.Node):
    cursor = tree.walk()
    visited_children = False

    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break


def strip_comments(content: bytes, root: ts.Node):
    spans = []
    for node in walk_tree(root):
        if node.type.endswith("comment") or node.type in ["haddock"]:
            spans.append((node.start_byte, node.end_byte))

    stripped = bytearray(content)

    for start, end in sorted(spans, reverse=True):
        stripped[start:end] = b" "

    return stripped


def main():
    ds = load_dataset(
        "bigcode/the-stack",
        streaming=True,
        split="train",
        data_files=[
            # You could add more if you want
            "data/python/train-00000-of-00206.parquet",
            "data/c/train-00000-of-00257.parquet",
            "data/java/train-00000-of-00285.parquet",
            "data/c++/train-00000-of-00214.parquet",
            "data/javascript/train-00000-of-00499.parquet",
            "data/go/train-00000-of-00115.parquet",
            "data/rust/train-00000-of-00040.parquet",
            "data/php/train-00000-of-00198.parquet",
            "data/haskell/train-00000-of-00007.parquet",
        ],
    )

    function_count = {language: 0 for language in SUPPORTED_LANGUAGES}
    trees = []

    for sample in tqdm(ds):
        if all(
            map(lambda count: count >= FUNCTIONS_PER_LANGUAGE, function_count.values())
        ):
            print("Data collection complete, saving...")
            print(function_count)
            break

        detected_language = detect_language_from_extension(sample["ext"])

        if not (language := SUPPORTED_LANGUAGES.get(detected_language)):
            continue

        if function_count[detected_language] >= FUNCTIONS_PER_LANGUAGE:
            continue

        content = bytes(sample["content"], "utf-8")

        tree = language.ts_parser.parse(content)
        stripped = strip_comments(content, tree)
        tree = language.ts_parser.parse(stripped)

        if not (
            functions := language.query_cursor.captures(tree.root_node).get("functions")
        ):
            continue

        function_count[detected_language] += len(functions)
        trees.extend(
            [
                Cst.from_parse_tree(
                    detected_language,
                    stripped[function.start_byte : function.end_byte],
                    function,
                )
                for function in functions
            ]
        )

    with open("trees.json", "w") as file:
        json.dump([tree.to_dict() for tree in trees], file)


if __name__ == "__main__":
    main()
