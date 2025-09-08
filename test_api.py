import requests

API_URL = "http://localhost:9012/predict"   # or https:// if using SSL
IMAGE_PATH = "/home/cuongph14/Downloads/Phương tiện truyền thông (1).jfif"                     # change to your test image

def test_predict():
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": f}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        print("✅ Success:", response.json())
    else:
        print("❌ Failed:", response.status_code, response.text)


if __name__ == "__main__":
    test_predict()