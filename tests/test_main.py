"""Tests for the main entry point."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from src.main import app, orchestrator


class TestCLI:
    """Test CLI functionality."""
    
    @pytest.mark.asyncio
    async def test_interactive_chat(self):
        """Test interactive chat functionality."""
        with patch('builtins.input', return_value="Hello\nquit"):
            with patch('builtins.print'):
                from src.main import interactive_chat
                await interactive_chat("test_user")
    
    @pytest.mark.asyncio
    async def test_run_scenario(self):
        """Test scenario execution."""
        scenario_data = [
            {"user_id": "test", "message": "Hello"},
            {"user_id": "test", "message": "How are you?"}
        ]
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "[]"
            with patch('yaml.safe_load', return_value=scenario_data):
                with patch('src.main.Orchestrator') as mock_orch:
                    mock_orch.return_value.run_turn = AsyncMock(return_value="Test response")
                    from src.main import run_scenario
                    await run_scenario("test.yaml")


class TestHTTPAPI:
    """Test HTTP API functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "orchestrator_ready" in data
    
    @pytest.mark.asyncio
    async def test_chat_endpoint(self):
        """Test chat endpoint."""
        with patch('src.main.orchestrator') as mock_orch:
            mock_orch.run_turn = AsyncMock(return_value="Test response")
            
            response = self.client.post("/chat", json={
                "user_id": "test_user",
                "message": "Hello"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Test response"
            assert data["user_id"] == "test_user"
    
    def test_chat_endpoint_invalid_request(self):
        """Test chat endpoint with invalid request."""
        response = self.client.post("/chat", json={
            "user_id": "test_user"
            # Missing message field
        })
        assert response.status_code == 422 