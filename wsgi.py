"""WSGI entry point used by gunicorn (``wsgi:application``)."""

from app import create_app

application = create_app()
