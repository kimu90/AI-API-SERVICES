#! /bin/bash

poetry run uvicorn ai_services_api.app:app --reload --host 0.0.0.0 --port 8000