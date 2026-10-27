"""Process-wide resources created in the application lifespan."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from compliance_core.config import Settings
from compliance_core.runtime import ServiceContext
from compliance_core.security.tokens import TokenSigner
from compliance_core.services.session_store import LoginThrottle, TokenRevocationList
from compliance_retrieval.config import RetrievalConfig


@dataclass(slots=True, kw_only=True)
class AppContext(ServiceContext):
    """The shared service resources plus the browser-session state only the API needs."""

    signer: TokenSigner
    throttle: LoginThrottle
    revocations: TokenRevocationList
    retrieval: RetrievalConfig

    @classmethod
    def create(cls, settings: Settings) -> AppContext:
        shared = ServiceContext.shared_resources(settings)
        redis = shared["redis"]
        return cls(
            **shared,
            signer=TokenSigner(
                secret=settings.resolved_jwt_secret(),
                issuer=settings.jwt_issuer,
                audience=settings.jwt_audience,
                ttl=timedelta(minutes=settings.jwt_access_ttl_minutes),
            ),
            throttle=LoginThrottle(
                redis, settings.login_max_failures, settings.login_failure_window_seconds
            ),
            revocations=TokenRevocationList(redis),
            retrieval=RetrievalConfig.from_settings(settings),
        )
