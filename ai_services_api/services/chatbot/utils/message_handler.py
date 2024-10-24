import logging
from typing import AsyncIterable
from .llm_manager import GeminiLLMManager

logger = logging.getLogger(__name__)

class MessageHandler:
    def __init__(self, llm_manager: GeminiLLMManager):
        self.llm_manager = llm_manager

    async def send_message_async(self, message: str) -> AsyncIterable[str]:
        """
        Sends the user message to the LLM and streams the response asynchronously.
        
        :param message: The user input message
        :return: Async generator that yields response tokens from the LLM
        """
        try:
            # Stream tokens from the LLM
            async for token in self.llm_manager.generate_async_response(message):
                yield token
        except Exception as e:
            logger.error(f"Error in send_message_async: {e}")
            raise RuntimeError("Failed to generate response from the LLM.") from e

    async def flush_conversation_cache(self, project_id: str):
        """
        Clears the conversation history stored in the memory for the given project ID.
        
        :param project_id: The unique identifier for the project/conversation
        """
        try:
            # Create or retrieve memory associated with the project ID
            history = self.llm_manager.create_or_get_memory(project_id)
            
            # Clear the conversation history from memory
            history.clear()
            logger.info(f"Successfully flushed conversation cache for project: {project_id}")
        except Exception as e:
            logger.error(f"Error while flushing conversation cache for project {project_id}: {e}")
            raise RuntimeError("Failed to clear conversation history.") from e