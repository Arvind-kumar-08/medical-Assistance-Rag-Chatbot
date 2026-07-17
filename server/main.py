from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middlewares.exception_handlers import catch_exception_middleware
from routes.upload_pdfs import router as upload_router
from routes.askquestion import router as ask_router

app=FastAPI(title="Medical Assistance Api",description="API for AI Medical Assistance chatbot")

#CORS setup

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# Middleware exception handlers 
app.middleware("http")(catch_exception_middleware)
# routers

# 1. Upload pdf document
app.include_router(upload_router)

# 2. Asking query
app.include_router(ask_router)
