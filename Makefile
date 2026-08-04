VERSION := $(shell python - <<'EOF'
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])
EOF
)

CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

# -----------------------------
# TestPyPI Release
# -----------------------------
release-test:
	@echo "Releasing to TestPyPI version $(VERSION)"
	@if git rev-parse "v$(VERSION)-test" >/dev/null 2>&1; then \
	    echo "ERROR: Tag v$(VERSION)-test already exists."; \
	    exit 1; \
	fi
	git tag v$(VERSION)-test
	git push origin v$(VERSION)-test

# -----------------------------
# Production PyPI Release
# -----------------------------
release-prod:
	@if [ "$(CURRENT_BRANCH)" != "main" ]; then \
	    echo "ERROR: Must be on main to release to PyPI."; \
	    exit 1; \
	fi
	@echo "Checking tag state for v$(VERSION)..."
	# Check local tag
	@if git rev-parse "v$(VERSION)" >/dev/null 2>&1; then \
	    local_exists=1; \
	else \
	    local_exists=0; \
	fi; \
	\
	# Check remote tag
	if git ls-remote --tags origin "v$(VERSION)" | grep "v$(VERSION)" >/dev/null; then \
	    remote_exists=1; \
	else \
	    remote_exists=0; \
	fi; \
	\
	# Case 1: local + remote → abort
	if [ $$local_exists -eq 1 ] && [ $$remote_exists -eq 1 ]; then \
	    echo "ERROR: Tag v$(VERSION) exists locally AND remotely. Aborting release."; \
	    exit 1; \
	fi; \
	\
	# Case 2: local only → delete local tag
	if [ $$local_exists -eq 1 ] && [ $$remote_exists -eq 0 ]; then \
	    echo "Local tag v$(VERSION) exists but remote does not. Deleting local tag..."; \
	    git tag -d v$(VERSION); \
	fi; \
	\
	# Case 3: create tag if needed
	if [ $$local_exists -eq 0 ]; then \
	    echo "Creating local tag v$(VERSION)"; \
	    git tag v$(VERSION); \
	fi; \
	\
	# Push tag if remote missing
	if [ $$remote_exists -eq 0 ]; then \
	    echo "Pushing tag v$(VERSION) to origin"; \
	    git push origin v$(VERSION); \
	else \
	    echo "Remote tag v$(VERSION) already exists. Skipping push."; \
	fi; \
	\
	# GitHub Release handling
	if gh release view v$(VERSION) >/dev/null 2>&1; then \
	    echo "GitHub Release v$(VERSION) already exists. Updating notes."; \
	    gh release edit v$(VERSION) --generate-notes; \
	else \
	    echo "Creating GitHub Release v$(VERSION)"; \
	    gh release create v$(VERSION) --generate-notes; \
	fi
	@echo "release-prod completed successfully."

# Optional alias
release: release-prod

# -----------------------------
# Version bumping
# -----------------------------
bump-patch:
	bump2version patch

bump-minor:
	bump2version minor

bump-major:
	bump2version major

# -----------------------------
# Tox
# -----------------------------
tox:
	tox

tox-lint:
	tox -e lint

# Remove this if no typecheck env exists
tox-typecheck:
	tox -e typecheck

# -----------------------------
# Version checks
# -----------------------------
verify-version:
	@echo "Checking version consistency..."
	@python scripts/verify-version.py

test-version:
	pytest -q tests/unit/test_version_consistency.py

.PHONY: \
    release-test release-prod release \
    bump-patch bump-minor bump-major \
    tox tox-lint tox-typecheck \
    verify-version test-version
