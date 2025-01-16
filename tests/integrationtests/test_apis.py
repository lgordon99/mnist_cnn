from fastapi.testclient import TestClient
from mnist_cnn.api import app
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    print(response)
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the MNIST model inference API!"}