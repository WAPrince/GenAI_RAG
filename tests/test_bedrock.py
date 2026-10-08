from pycharmai.aws.bedrock import BedrockClient


def test():
    client = BedrockClient("eu-central-1")
    response = client.converse(
        model_id="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        prompt="what date is today?",
    )
    answer = response["output"]["message"]["content"][0]["text"]
    print(answer)
    assert answer is not None
    assert len(answer) > 0
