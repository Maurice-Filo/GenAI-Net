"""Transport-only adapter for archived runners; no provider configuration is written.

Archived runners pin a provider/model (and sometimes a localhost endpoint). This
adapter replaces only those routing arguments. The configured headless Harness
profile owns credentials, endpoint, context size, sampling and reasoning settings.
"""
from functools import wraps
import hashlib
from pathlib import Path


def install(client_class, provider, model):
    original = client_class.__init__

    @wraps(original)
    def configured(self, *args, **kwargs):
        kwargs.update(provider=provider, model=model, openai_compatible_base_url=None)
        original(self, *args, **kwargs)

    client_class.__init__ = configured


def install_manifest_metadata(store, provider, model):
    """Label new run manifests before their identities are computed.

    The candidate-cache fingerprint path is unchanged. Only a harness campaign
    manifest has all three of schema, config and harness=True.
    """
    original = store.fingerprint

    @wraps(original)
    def fingerprint(value):
        if isinstance(value, dict) and value.get('harness') is True and 'schema' in value and 'config' in value:
            if 'reference_transport' not in value:
                value['reference_transport'] = {key: value.pop(key) for key in (
                    'model', 'provider', 'openai_compatible_base_url', 'local_context_window',
                    'local_max_output_tokens', 'local_reasoning') if key in value}
            value.update(model=model, provider=provider, model_configuration='user-owned Harness headless profile',
                         transport_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return original(value)

    store.fingerprint = fingerprint
