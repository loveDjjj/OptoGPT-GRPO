from our_work._shared.physics.structure import split_structure_token, tokens_to_tmm_config


def test_tokens_to_tmm_config_converts_nm_to_um():
    material, thickness = split_structure_token("SiO2_120")
    assert material == "SiO2"
    assert thickness == 120.0

    config = tokens_to_tmm_config(["SiO2_120", "Ge_250"], database_path="database")
    assert config["materials"] == ["SiO2", "Ge"]
    assert config["thicknesses"] == [0.12, 0.25]
    assert config["database_path"] == "database"
