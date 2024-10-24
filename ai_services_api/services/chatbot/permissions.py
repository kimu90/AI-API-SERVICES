from fastapi import Request, HTTPException
from .config import Settings

# Create an instance of the Settings class to access environment variables
settings = Settings()

# async def verify_api_key(request: Request):
#     api_key = request.headers.get("x-api-key")
#     expected_api_key = settings.MY_API_KEY  # Access MY_API_KEY from the instance

#     if not api_key:
#         raise HTTPException(status_code=401, detail="No API Key provided")
#     if api_key != expected_api_key:
#         raise HTTPException(status_code=403, detail="Invalid API Key")
    
#     return api_key