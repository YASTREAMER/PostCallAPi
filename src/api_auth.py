import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api_key_config import load_api_key


_EXPECTED_API_KEY = load_api_key()
_BEARER = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_BEARER),
    ],
) -> None:
    supplied_key = credentials.credentials if credentials is not None else ""
    valid = (
        credentials is not None
        and credentials.scheme.casefold() == "bearer"
        and secrets.compare_digest(supplied_key, _EXPECTED_API_KEY)
    )
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
