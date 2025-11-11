# app/__init__.py
"""
Слой приложения - управление жизненным циклом и сессиями
"""
from .application import StencilAnalyzerApplication
from .session_manager import SessionManager

__all__ = ['StencilAnalyzerApplication', 'SessionManager']