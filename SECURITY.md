# Security

tetolate is experimental software intended for trusted, small-scale deployments. The
web interface is protected by a single administrator account, not a hardened
multi-tenant account system. Do not expose it directly to the internet without TLS, a
reverse proxy, resource limits, and appropriate network controls.

## Reporting a Vulnerability

Once the repository is hosted on GitHub, report vulnerabilities with a private GitHub
security advisory. Do not include credentials, copyrighted source pages, job archives,
or unredacted VLM traces in a public issue.

## Credentials and Job Data

- Copy the checked-in `*.example.json` files to local configuration files and set their
  permissions to `0600`.
- Active JSON files under `data/config/` are ignored; only `*.example.json`
  templates are committed. Verify `git status` before every commit rather than
  relying only on ignore rules.
- Rotate any credential that was committed, pasted into an issue, or included in a
  shared debug archive. Removing it from the latest commit is not sufficient.
- Job directories and debug traces can contain source images, OCR text, translations,
  model prompts, model reasoning, endpoint URLs, and category names. Treat the
  complete output directory as private.
- API keys and passwords are intentionally absent from all checked-in examples. The
  first server startup generates an admin password and persists only its salted hash in
  `.tetolate-web-state.json` under the configured jobs directory.

The Docker configuration listens on all container interfaces, but Compose
publishes it only on host loopback. Listening on a public host interface is an
explicit deployment decision and should be protected by a reverse proxy.
Run only one web process for each configured jobs directory. tetolate rejects a second
process using the same directory; its authenticated sessions and queue coordination are
not designed for multi-worker deployment.

Input CBZ files are checked for unsafe names, excessive entry counts, expanded archive
size, preserved metadata size, and decoded image dimensions. These are safety limits,
not a complete substitute for container resource limits.
