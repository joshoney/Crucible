import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from worker import main

@pytest.mark.asyncio
async def test_worker_initialization():
    with patch("worker.Client.connect") as mock_connect, \
         patch("worker.Worker") as mock_worker_cls:
        
        mock_client = AsyncMock()
        mock_connect.return_value = mock_client
        
        mock_worker_instance = MagicMock()
        mock_worker_instance.run = AsyncMock()
        mock_worker_cls.return_value = mock_worker_instance
        
        # Act
        await main()
        
        # Assert
        mock_connect.assert_called_once()
        mock_worker_cls.assert_called_once()
        mock_worker_instance.run.assert_called_once()
        
        # Check that activities and workflows were passed
        _, kwargs = mock_worker_cls.call_args
        assert "eval-task-queue" == kwargs["task_queue"]
        assert len(kwargs["workflows"]) == 1
        assert len(kwargs["activities"]) == 3
