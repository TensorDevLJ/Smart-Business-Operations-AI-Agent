"""
Database Seeding Script.

Generates and inserts realistic synthetic business data.
Run this BEFORE training models.

Usage:
    python scripts/seed_database.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.database.seed_data import seed_database

if __name__ == "__main__":
    seed_database()
