from conftest import write_xml
from panoramic_features.annotations import parse_annotation


def test_parse_polygons(tmp_path):
    square = [(1, 2), (10, 2), (10, 9), (1, 9)]
    write_xml(tmp_path / "a.xml", {"48": square, "Sağ M3": square[:3]})
    result = parse_annotation(tmp_path / "a.xml")
    assert result["48"] == [square]
    assert len(result["Sağ M3"][0]) == 3


def test_degenerate_polygons_are_ignored(tmp_path):
    write_xml(tmp_path / "a.xml", {"38": [(1, 1), (2, 2)]})
    assert parse_annotation(tmp_path / "a.xml") == {}
