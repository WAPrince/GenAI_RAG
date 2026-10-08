from pycharmai.aws.identity import get_identity


def test_aws_identity():
    identity = get_identity()
    assert "Account" in identity
    assert "Arn" in identity