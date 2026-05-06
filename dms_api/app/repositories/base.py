"""
Base Repository

Abstract base class for repository implementations.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseRepository(ABC):
    """
    Abstract base repository.

    Repositories handle data access and external service communication.
    """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the underlying service is healthy."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources."""
        pass
