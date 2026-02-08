"""
Repository Layer for Data Access.

This module provides the repository pattern implementation for MedAI Backend,
abstracting all database operations and providing a clean interface for
data access.

Repositories:
    - EpisodeRepository: Handles episode and note persistence in MongoDB
"""

from app.repositories.episode_repository import EpisodeRepository

__all__ = ["EpisodeRepository"]
