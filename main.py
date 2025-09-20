import os
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    a = 5
    print(f"value ${a}")
    return {"message": "Hello!"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    print(f"Running on port {port} with reload")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
