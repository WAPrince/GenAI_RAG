from pycharmai.aws.bedrock import BedrockClient


def test():
    client = BedrockClient("us-east-1")
    response = client.converse(
        model_id="us.anthropic.claude-sonnet-4-6",
        prompt="what date is today?",
    )
    answer = response["output"]["message"]["content"][0]["text"]
    print(answer)
    assert answer is not None
    assert len(answer) > 0
