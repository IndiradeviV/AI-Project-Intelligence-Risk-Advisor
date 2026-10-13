
from ingestion.document_loader import extract_text


def test_txt_extraction():
    content = b"Project Name: WeatherGPT\nStatus: In Progress"

    result = extract_text("sample.txt", content)

    assert "WeatherGPT" in result
    assert "In Progress" in result

    print("TXT extraction successful!")
    print(result)


if __name__ == "__main__":
    test_txt_extraction()
