import xlearn.datasets.processors as proc

def test_key_mapper():
    data_raw = {'a': 1, 'b': 2}
    key_out = ['o1', 'o2']
    key_map = {'o1': 'a', 'o2': 'a'}
    pass_keys = ['o1']
    km = proc.get_instance('proc', keys_out=key_out, keys_map=key_map, pass_keys=pass_keys)
    # km = proc.Proc(keys_out=key_out, keys_map=key_map, pass_keys=pass_keys)
    data_post = km(data_raw)
    assert data_post['o1'] == 1
    assert data_post['o2'] == 1