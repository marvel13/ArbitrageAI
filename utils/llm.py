"""Generic Gemini LLM client — transport layer only.

Wraps the google-genai SDK to call Gemini with JSON-mode schema enforcement.
Includes a simple retry loop. Contains NO phase-specific prompts, schemas,
or post-processing logic. Each pipeline phase brings its own.
"""

import json
import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Load .env for configuration
load_dotenv()

MAX_RETRIES = 3

class GeminiClient:
    """Thin, reusable wrapper around google-genai for structured JSON calls.

    Usage by any phase agent::

        client = GeminiClient()
        raw = client.generate_json(
            system_prompt="You are ...",
            user_prompt="Analyze ...",
            schema=MyPydanticModel.model_json_schema(),
            model="gemini-2.0-flash",
        )
        result = MyPydanticModel.model_validate(raw)
    """

    def __init__(self) -> None:
        """Initialize the Vertex AI Gemini client.

        Reads configuration from environment variables:
            GOOGLE_APPLICATION_CREDENTIALS - path to service account JSON
            GOOGLE_CLOUD_PROJECT - GCP project ID
            GOOGLE_CLOUD_LOCATION - GCP region
        """
        credentials_path = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "vertex-ai-credentials.json",
        )
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "nyayanidhi-main")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Import here so the module stays lightweight if unused
        from google.oauth2 import service_account

        self._client = genai.Client(
            vertexai=True,
            credentials=service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            ),
            project=project,
            location=location,
        )
        logger.info("GeminiClient initialized (project=%s, location=%s)", project, location)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.2,
    ) -> dict:
        """Send a prompt to Gemini and return parsed JSON.

        Enforces JSON-mode output via the provided schema. Retries up
        to MAX_RETRIES times with exponential backoff on any failure.

        Args:
            system_prompt: System-level instruction for the LLM.
            user_prompt: The user/content prompt.
            schema: JSON schema dict (e.g. from Pydantic model_json_schema()).
            model: Gemini model name to use.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            start = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_json_schema=schema,
                        temperature=temperature,
                    ),
                )
                elapsed = time.perf_counter() - start

                raw_json = json.loads(response.text)

                logger.info(
                    "LLM call succeeded in %.1fs (model=%s, attempt %d)",
                    elapsed,
                    model,
                    attempt,
                )
                return raw_json

            except Exception as exc:
                elapsed = time.perf_counter() - start
                last_error = exc
                logger.warning(
                    "Attempt %d/%d failed (model=%s, %.1fs): %s",
                    attempt,
                    MAX_RETRIES,
                    model,
                    elapsed,
                    exc,
                )
                if attempt < MAX_RETRIES:
                    sleep_time = 2**attempt
                    logger.info("Retrying in %ds...", sleep_time)
                    time.sleep(sleep_time)

        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES} attempts: {last_error}"
        )
