# Hybrid Setup (Fast Local ARM)

Use Docker only for infrastructure (`db`, `redis`) and run Django/ML locally on your machine.

## Start Infra

```bash
scripts/hybrid-up.sh
```

## Run Django Commands Locally

```bash
scripts/hybrid-manage.sh migrate
scripts/hybrid-manage.sh check
```

## Run ML Pipeline Command Locally

```bash
scripts/hybrid-manage.sh run_ml_sample_to_db --session-uuid <SESSION_UUID>
```

## Optional Local Celery Worker

```bash
scripts/hybrid-celery.sh
```

## ADC Expectations

- Host ADC file should exist at:
  - `$HOME/.config/gcloud/application_default_credentials.json`
- Ensure `.env` contains:
  - `GOOGLE_CLOUD_PROJECT=<your-project-id>`

## Stop Infra

```bash
scripts/hybrid-down.sh
```

## Rollback (if needed)

From repo root:

```bash
git restore SETUP.md scripts/
```

This reverts only the hybrid helper/docs layer and does not touch ML logic.
