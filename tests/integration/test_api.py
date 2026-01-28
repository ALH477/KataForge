"""
KataForge - Adaptive Martial Arts Analysis System
Copyright © 2026 DeMoD LLC. All rights reserved.

This file is part of KataForge, released under the KataForge License
(based on Elastic License v2). See LICENSE in the project root for full terms.

SPDX-License-Identifier: Elastic-2.0

Description:
    [Brief module description – please edit]

Usage notes:
    - Private self-hosting, dojo use, and modifications are permitted.
    - Offering as a hosted/managed service to third parties is prohibited
      without explicit written permission from DeMoD LLC.
"""

import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_api_health_check():
    async with AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "checks" in data

@pytest.mark.asyncio
async def test_csrf_protection():
    async with AsyncClient(base_url="http://localhost:8000") as client:
        # Get CSRF token
        resp = await client.get("/health/live")
        token = resp.cookies.get("csrf_token")

        # Try protected endpoint without token
        response = await client.post("/secure-endpoint")
        assert response.status_code == 403

        # Valid request with token
        response = await client.post(
            "/secure-endpoint",
            headers={"X-CSRFToken": token}
        )
        assert response.status_code == 200