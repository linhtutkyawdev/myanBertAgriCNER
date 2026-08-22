from mycner.data.converter import serialize_mt5

def test_serialize_mt5_normal():
    tokens = ["စပါး", "ရာတွင်", "ယူရီးယား"]
    labels = ["CROP", "O", "FERT"]
    res = serialize_mt5(tokens, labels)
    assert res["input_text"] == "cner: စပါး ရာတွင် ယူရီးယား"
    assert res["target_text"] == "စပါး[CROP] ရာတွင် ယူရီးယား[FERT]"
