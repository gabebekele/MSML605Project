from src.validation import validate_threshold

def test_valid_threshold():
    validate_threshold(0.5)

def test_invalid_threshold():
    try:
        validate_threshold(2)
        assert False
    except:
        assert True