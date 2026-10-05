from pkg import Client


def test_run():
    assert Client().run() == 2
