# crud_redis.py

import json
import numpy as np
import redis
import logging

logger = logging.getLogger(__name__)

class RedisCRUD:
    def __init__(self, redis_url=None):
        if redis_url is None:
            redis_url = "redis://localhost:6379"
        self.redis_client = redis.StrictRedis.from_url(redis_url, decode_responses=True)

    def store_redis_data(self, embedding, text):
        """Stores a new text and its corresponding embedding in Redis."""
        key = f"embedding:{hash(text)}"  # Unique key for each text
        self.redis_client.hmset(key, {"embedding": json.dumps(embedding), "text": text})
        logger.info(f"Data stored in Redis with key: {key}")
        return key  # Return the key in case we need it later

    def query_redis_data(self, query_vector, top_n=3, threshold=0.5):
        """Fetch relevant data points from Redis based on the user query vector."""
        keys = self.redis_client.keys("embedding:*")
        scores = []

        for key in keys:
            embedding_str = self.redis_client.hget(key, "embedding")
            embedding = np.array(json.loads(embedding_str))

            # Calculate cosine similarity
            score = np.dot(embedding, query_vector) / (np.linalg.norm(embedding) * np.linalg.norm(query_vector))
            scores.append((score, self.redis_client.hget(key, "text")))

        # Sort by score and select the top N above the threshold
        scores.sort(key=lambda x: x[0], reverse=True)
        relevant_texts = [text for score, text in scores if score >= threshold][:top_n]
        logger.debug(f"Retrieved {len(relevant_texts)} relevant texts from Redis.")
        return relevant_texts

    def update_redis_data(self, key, new_embedding, new_text):
        """Updates existing text and its corresponding embedding in Redis."""
        if self.redis_client.exists(key):
            self.redis_client.hmset(key, {"embedding": json.dumps(new_embedding), "text": new_text})
            logger.info(f"Data updated in Redis for key: {key}")
        else:
            logger.warning(f"Key {key} does not exist in Redis. Cannot update.")

    def delete_redis_data(self, key):
        """Deletes an embedding and text from Redis."""
        if self.redis_client.exists(key):
            self.redis_client.delete(key)
            logger.info(f"Data with key {key} deleted from Redis.")
        else:
            logger.warning(f"Key {key} does not exist in Redis. Cannot delete.")
