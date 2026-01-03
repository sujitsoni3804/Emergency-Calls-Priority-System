from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socket
import uvicorn
from scripts.routes import router
from scripts.database import init_db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

app.include_router(router)

if __name__ == '__main__':
    print("Local URL: http://127.0.0.1:5000")
    print(f"Local (LAN) URL: http://{socket.gethostbyname(socket.gethostname())}:5000")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)