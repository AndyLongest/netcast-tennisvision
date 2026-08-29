# Security policy

Netcast TennisVision currently binds only to `127.0.0.1` and is designed for one trusted local
operator. Do not expose `server.py` directly to a LAN or the internet: it has no accounts,
authentication, tenant isolation or hardened upload sandbox.

## Sensitive data

- Uploaded match videos and generated reports remain under `data/` and are ignored by Git.
- Never attach user videos, logs containing user paths, or generated reports to a public
  issue without explicit permission.
- Do not commit tokens, credentials, cookies or local environment files.

## Model safety

PyTorch and pickle model files are loaded only after comparing their SHA-256 against
frozen constants/manifests. Treat replacement `.pt` and `.pkl` files as executable supply
chain inputs; never load an unverified asset.

## Reporting

Until a public maintainer contact is selected, report security concerns privately to the
repository owner. Add that contact before publication.
