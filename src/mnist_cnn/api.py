from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the MNIST model inference API!"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    """Get an item by id."""
    return {"item_id": item_id}