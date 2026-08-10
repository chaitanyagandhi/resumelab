"""Construction of the configured LLM client.

The pipeline depends on the :class:`~resumelab.llm.client.LLMClient` protocol; this
is the one place that knows which concrete adapter implements it. Adding a provider
means adding an adapter and one entry here.
"""

from __future__ import annotations

import logging

from resumelab.config import LLMProvider, Settings
from resumelab.llm.anthropic_client import AnthropicClient
from resumelab.llm.client import LLMClient
from resumelab.llm.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def create_llm_client(
    settings: Settings,
    *,
    provider: LLMProvider | None = None,
) -> LLMClient:
    """Build the client for ``provider``.

    Args:
        settings: Loaded application settings.
        provider: Provider to use, overriding configuration. Supplied by the CLI when
            a researcher chooses a provider for a single run.

    Returns:
        A client implementing :class:`LLMClient`.

    Raises:
        ConfigurationError: If the selected provider has no API key configured.
    """
    selected = provider if provider is not None else settings.resolved_provider
    # Fails here, before any network call, when the key for this provider is missing.
    settings.api_key_for(selected)

    client: LLMClient = (
        OpenAIClient(settings) if selected is LLMProvider.OPENAI else AnthropicClient(settings)
    )
    logger.info("using llm provider=%s model=%s", selected.value, client.model)
    return client
