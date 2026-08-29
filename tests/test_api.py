from fastapi.testclient import TestClient

from api.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "patch": (
                    "diff --git a/example.py b/example.py\n"
                    "--- a/example.py\n"
                    "+++ b/example.py\n"
                    "@@ -1,1 +1,1 @@\n"
                    "-return balance[user_id]\n"
                    "+return balance.get(user_id, 0)"
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in {
        "bug_fix",
        "non_bug_fix",
    }

    assert 0.0 <= data["bug_fix_probability"] <= 1.0
    assert 0.0 <= data["non_bug_fix_probability"] <= 1.0
