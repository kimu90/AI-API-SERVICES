# In app/controllers/chatbot_router.py

from fastapi import APIRouter
from ai_services_api.services.chatbot.app.api.endpoints.v1 import conversation

api_router = APIRouter()

# Include the conversation router
api_router.include_router(
    conversation.router,
    prefix="/conversation",  # Prefix for conversation endpoints
    tags=["conversation"]  # Tag for documentation
)


