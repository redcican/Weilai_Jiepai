"""
Base Service

Abstract base class for service implementations.
"""

from abc import ABC
from ..repositories.dms import DMSRepository
from ..core.logging import get_logger


class BaseService(ABC):
    """
    Abstract base service.

    Services contain business logic and orchestrate repository calls.
    """

    def __init__(self, repository: DMSRepository):
        self._repository = repository
        self._logger = get_logger(self.__class__.__name__)
