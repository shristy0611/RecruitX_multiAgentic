import pytest
from fastapi.testclient import TestClient

from recruitx_app.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint returns the expected response."""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong!"} 