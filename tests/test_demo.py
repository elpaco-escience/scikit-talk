from sktalk.demo import Demo


def test_demo():
    person = Demo("Person")
    assert person.name == 'Person'
