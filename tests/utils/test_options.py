import nose
import tempfile
import json
import xlearn.utils.options as o

def test_json_config_normal_call():
        @o.json_config
        def foo(a, b, *, c, d, e=4):
            assert [a, b, c, d, e] == list(range(5))
        foo(0, 1, c=2, d=3)

def test_json_config_file():
    with tempfile.NamedTemporaryFile(mode='w+') as fp:
        fn = fp.name
        a = {'b': 2, 'd': 4}
        json.dump(a, fp)
        fp.seek(0)
        a_r = json.load(fp)        
        @o.json_config
        def foo(a, b, *, c, d, e=5):
            assert [a, b, c, d, e] == list(range(1, 6))            
        foo(1, c=3, config_files=fn)

def test_json_config_redundent():
    with tempfile.NamedTemporaryFile(mode='w+') as fp:
        fn = fp.name
        a = {'b': 2, 'd': 4, 'x': 100}
        json.dump(a, fp)
        fp.seek(0)
        a_r = json.load(fp)        
        @o.json_config
        def foo(a, b, *, c, d, e=5):
            assert [a, b, c, d, e] == list(range(1, 6))            
        foo(1, c=3, config_files=fn)

def test_auto_configs():
    class A:
        @o.auto_configs()
        def __init__(self, a, b=None, c=2):
            pass
    a = A(4, 5, 6)
    assert a.c.a == 4
    assert a.c.b == 5
    assert a.c.c == 6

def test_auto_configs_defaults():
    class A:
        @o.auto_configs()
        def __init__(self, a, b=None, c=2):
            pass
    a = A(4, 5)
    assert a.c.a == 4
    assert a.c.b == 5
    assert a.c.c == 2

def test_auto_configs_exclude():
    class A:
        @o.auto_configs(exclude=['b'])
        def __init__(self, a, b=None, c=2):
            pass
    a = A(4, 5, 6)
    assert a.c.a == 4
    assert not a.c.has_key('b')
    assert a.c.c == 6
    
def test_auto_configs_include():
    class A:
        @o.auto_configs(include=['a'], exclude=['b'])
        def __init__(self, a, b=None, c=2):
            pass
    a = A(4, 5, 6)
    assert a.c.a == 4
    assert not a.c.has_key('b')
    assert not a.c.has_key('c')
    

def test_auto_configs_inherit():
    class A:
        @o.auto_configs()
        def __init__(self, a, b=None, *,  c=2):
            pass
    
    class B(A):
        @o.auto_configs()
        def __init__(self, d, *args, e, **kwargs):
            super(B, self).__init__(*args, **kwargs)
    b = B(3, a=0, b=1, c=2, e=4)
    assert b.c.a == 0
    assert b.c.b == 1
    assert b.c.c == 2
    assert b.c.d == 3
    assert b.c.e == 4

def test_combine_auto_json_config():
    class A:
        @o.json_config
        @o.auto_configs()
        def __init__(self, a, b=None, *, c, d=100):
            pass
    
    class B(A):
        @o.json_config
        @o.auto_configs()
        def __init__(self, a, b, c, d, e):
            super(B, self).__init__(a, b, c=c, d=d)

    with tempfile.NamedTemporaryFile(mode='w+') as fp:
        fn = fp.name
        a = {'a': 100, 'b': 1, 'd': 3, 'c': 2, 'x': 100}        
        json.dump(a, fp)
        fp.seek(0)
        a_r = json.load(fp)        
        b = B(a=0, e=4, config_files=fn)
        assert b.c.a == 0
        assert b.c.b == 1
        assert b.c.c == 2
        assert b.c.d == 3
        assert b.c.e == 4


