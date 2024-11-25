from pydantic import BaseModel

from document_ai_agents.schema_utils import (
    delete_keys_recursive,
    prepare_schema_for_gemini,
    replace_value_in_dict,
)


class MyModel(BaseModel):
    name: str
    value: int
    nested: dict = {}


def test_replace_value_in_dict_simple():
    test_data = {"a": 1, "b": {"$ref": "#/definitions/x"}}
    original_schema = {"definitions": {"x": 2}}
    assert replace_value_in_dict(test_data, original_schema) == {"a": 1, "b": 2}


def test_replace_value_in_dict_nested():
    test_data = {"a": 1, "b": {"c": {"$ref": "#/definitions/x"}}}
    original_schema = {"definitions": {"x": 2}}
    assert replace_value_in_dict(test_data, original_schema) == {"a": 1, "b": {"c": 2}}


def test_replace_value_in_dict_list():
    test_data = [1, {"$ref": "#/definitions/x"}]
    original_schema = {"definitions": {"x": 2}}
    assert replace_value_in_dict(test_data, original_schema) == [1, 2]


def test_delete_keys_recursive_dict():
    test_data = {"a": 1, "b": {"c": 2, "d": 3}}
    delete_keys_recursive(test_data, "c")
    assert test_data == {"a": 1, "b": {"d": 3}}


def test_delete_keys_recursive_list():
    test_data = [{"a": 1}, {"b": 2, "c": 3}]
    delete_keys_recursive(test_data, "c")
    assert test_data == [{"a": 1}, {"b": 2}]


def test_delete_keys_recursive_nested():
    test_data = {"a": 1, "b": {"c": 2, "d": {"e": 4, "c": 5}}}
    delete_keys_recursive(test_data, "c")
    assert test_data == {"a": 1, "b": {"d": {"e": 4}}}


def test_prepare_schema_for_gemini_basic():
    model = MyModel(name="test", value=123)
    schema = prepare_schema_for_gemini(model)
    assert "$defs" not in schema
    assert "title" not in schema
    assert "default" not in schema
