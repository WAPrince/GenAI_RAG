from langchain_core.tools import tool


@tool
def delete_file(filename: str) -> str:
    """Delete a file from the application workspace."""

    return f"File '{filename}' deleted."