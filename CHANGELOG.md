# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Deprecated

- `creyone_model.factory.create_model` is deprecated and will be **removed in v1.1.0**.
  Calling it now raises a `DeprecationWarning`. Use
  `creyone_model.factory.get_config(task_name)().create_model(...)` instead
  (e.g. `ModelCfg().create_model(...)` or the task-specific config's
  `create_model` method).
