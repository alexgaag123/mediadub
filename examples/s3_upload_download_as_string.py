import os

from media_dub.common.s3_sevice import load_text_from_s3, save_text_to_s3

if __name__ == "__main__":
    string_to_translate = "Hello! Could you please translate me?"

    save_text_to_s3("english-text", "test.txt", string_to_translate)

    downloaded_str = load_text_from_s3("english-text", "test.txt")

    print(downloaded_str)
